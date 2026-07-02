/**
 * src/memory/write.ts
 *
 * Write path: extract → embed → dedupe → store.
 *
 * extractMemories — calls qwen-plus to pull typed memories from a conversation turn.
 * writeMemories   — embeds candidates, deduplicates, and persists atomically.
 *
 * Conflict/supersession is handled in the NEXT prompt (sleep.ts).
 * Server-authoritative: all mutation is here; clients send only turn text.
 */

import { z } from "zod";
import { chatJSON, embed } from "../llm/qwen.js";
import { config } from "../config.js";
import {
  findByExactContent,
  insertMemory,
  touchMemory,
  vecSearch,
  type MemoryType,
} from "../db/queries.js";
import type { Db } from "../db/client.js";
import { detectSupersession } from "./sleep.js";

// ─── Extracted memory shape (pre-write, no id/user_id yet) ───────────────────

export interface ExtractedMemory {
  type: MemoryType;
  content: string;
  salience: number;
  confidence: number;
  tags: string[];
}

// ─── Candidate (extracted memory with optional pre-computed embedding) ────────
// Accepting embedding lets callers (and tests) skip the embed API call.

export interface MemoryCandidate extends ExtractedMemory {
  embedding?: number[];
}

// ─── Write result ─────────────────────────────────────────────────────────────

export type SkipReason = "exact_match" | "semantic_duplicate";

export interface WriteResult {
  inserted: string[];                                        // new memory ids
  skipped: { matched_id: string; reason: SkipReason }[];
}

// ─── Zod schema for chatJSON validation ──────────────────────────────────────

const ExtractedMemorySchema = z.object({
  type: z.enum(["fact", "preference", "decision", "event", "skill", "note"]),
  content: z.string().min(5),
  salience: z.number().min(0).max(1),
  confidence: z.number().min(0).max(1),
  tags: z.array(z.string()),
});

const ExtractionResponseSchema = z.object({
  memories: z.array(ExtractedMemorySchema),
});

/** Extraction cap per turn (SPEC §4.6). */
const MAX_MEMORIES_PER_TURN = 6;

// ─── System prompt ────────────────────────────────────────────────────────────

const EXTRACTION_SYSTEM = `\
You are Engram's memory extraction engine. Given a conversation turn (and optionally the assistant's response), extract every piece of information worth remembering across future sessions.

RULES:
1. Each memory must be ATOMIC (one fact per item) and SELF-CONTAINED — no pronouns; include the full subject ("The user prefers…" not "They prefer…").
2. Write in third person: "The user prefers X", "The user works as Y", "The user decided Z".
3. SKIP: small talk, filler, questions without answers, one-time ephemeral requests.
4. INCLUDE: stable facts, preferences, recurring decisions, skills, notable events.
5. NEVER invent facts that are not present in the turn — extract only what was actually said.
6. Return AT MOST 6 memories per turn; pick the most decision-relevant ones.

TYPES — choose the most specific:
  fact        objective information about the user or the world
  preference  how the user likes things (UI, workflow, food, tone…)
  decision    a choice the user explicitly made
  event       something notable that happened
  skill       something the user knows how to do
  note        anything else worth persisting

SALIENCE (0–1): importance for future decisions.
  preferences → 0.7–0.9 | facts → 0.4–0.7 | events → 0.2–0.5

CONFIDENCE (0–1): certainty the statement is true.
  explicit statement → 0.9–1.0 | inferred → 0.5–0.75

Return ONLY valid JSON matching this schema (no markdown, no prose):
{"memories":[{"type":"...","content":"...","salience":0.0,"confidence":0.0,"tags":["..."]}]}

If nothing is worth remembering, return: {"memories":[]}`;

// ─── extractMemories ──────────────────────────────────────────────────────────

/**
 * Call qwen-plus to extract typed memories from a user turn and optional
 * assistant decision. Returns [] if nothing is memorable.
 */
export async function extractMemories(
  userTurn: string,
  assistantDecision?: string
): Promise<ExtractedMemory[]> {
  const userContent = assistantDecision
    ? `[USER TURN]\n${userTurn}\n\n[ASSISTANT RESPONSE]\n${assistantDecision}`
    : `[USER TURN]\n${userTurn}`;

  const result = await chatJSON(
    {
      system: EXTRACTION_SYSTEM,
      messages: [{ role: "user", content: userContent }],
      model: config.ENGRAM_EXTRACT_MODEL,
      temperature: 0.1,   // low temperature for consistent structured output
    },
    ExtractionResponseSchema
  );

  // Hard cap regardless of what the model returned (rule 6 is advisory to it).
  return result.memories.slice(0, MAX_MEMORIES_PER_TURN);
}

// ─── writeMemories ────────────────────────────────────────────────────────────

/**
 * For each candidate:
 *   1. Exact content-hash check (free) → skip + touch if already active.
 *   2. Embed (if not pre-provided).
 *   3. Semantic dedup via vecSearch: if nearest cosine distance ≤ (1 - threshold) → skip + touch.
 *   4. Insert into memories + vec_memories with session_id provenance
 *      (atomic transaction via insertMemory).
 *   5. Write-time supersession: batched qwen verdict over the top-5 nearest;
 *      superseded olds get a `supersedes` edge + belief_event (SPEC §4.6).
 *
 * Returns ids of inserted memories and details on skipped ones.
 */
export async function writeMemories(
  userId: string,
  sessionId: string | null,
  candidates: MemoryCandidate[],
  db: Db
): Promise<WriteResult> {
  const result: WriteResult = { inserted: [], skipped: [] };
  // cosine distance threshold: similarity ≥ 0.95 → distance ≤ 0.05
  const distanceThreshold = 1 - config.ENGRAM_DEDUPE_THRESHOLD;

  for (const candidate of candidates) {
    // ── 1. Exact content match (fast path, no embedding needed) ──────────────
    const exact = findByExactContent(db, userId, candidate.content);
    if (exact) {
      touchMemory(db, exact.id);
      result.skipped.push({ matched_id: exact.id, reason: "exact_match" });
      continue;
    }

    // ── 2. Embed (or use pre-provided vector) ─────────────────────────────────
    let embedding: number[];
    if (candidate.embedding) {
      embedding = candidate.embedding;
    } else {
      const [vec] = await embed([candidate.content]);
      embedding = vec;
    }

    // ── 3. Semantic dedup ─────────────────────────────────────────────────────
    // vecSearch uses cosine distance (distance_metric=cosine in schema).
    // Over-fetch 5; we only care about the nearest match.
    const neighbours = vecSearch(db, userId, embedding, 5);
    const nearest = neighbours[0];
    if (nearest && nearest.distance <= distanceThreshold) {
      touchMemory(db, nearest.id);
      result.skipped.push({ matched_id: nearest.id, reason: "semantic_duplicate" });
      continue;
    }

    // ── 4. Store (with session provenance) ────────────────────────────────────
    const id = insertMemory(db, {
      user_id: userId,
      type: candidate.type,
      content: candidate.content,
      embedding,
      salience: candidate.salience,
      confidence: candidate.confidence,
      source: "extracted",
      session_id: sessionId,
      tags: candidate.tags,
    });

    result.inserted.push(id);

    // ── 5. Write-time supersession (best-effort; non-fatal) ───────────────────
    try {
      await detectSupersession(
        userId,
        { id, type: candidate.type, content: candidate.content, embedding },
        db,
        sessionId,
      );
    } catch {
      // Supersession failure never blocks the write
    }
  }

  return result;
}
