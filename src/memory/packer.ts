/**
 * src/memory/packer.ts
 *
 * Token-budget context packer (SPEC §4.3).
 *
 * packContext(candidates, budgetTokens):
 *   1. RESERVED PHASE — a sub-budget (ENGRAM_PACK_RESERVED_FRACTION of B,
 *      default 25%) is filled first with critical memories: pinned +
 *      preference/decision, by utility (pinned = Infinity, so always first).
 *   2. GREEDY PHASE — every remaining candidate (including critical overflow
 *      from phase 1) competes by utility for the rest of the budget.
 *   3. Overflow high-utility item → qwen-flash compression ("rewrite ≤N tokens,
 *      preserve decision-relevant facts"), one retry with a tighter target.
 *   4. A pinned item that still can't fit is deterministically truncated —
 *      the budget is a HARD guarantee (asserted), pinned presence is second.
 *
 * Returns { text, manifest[{id, type, tokens, utility, session_born}], totalTokens }
 * with totalTokens ≤ budget always.
 *
 * Token counting: tiktoken o200k_base as an estimator (Qwen uses its own
 * tokenizer; o200k is a close, deterministic proxy).
 */

import assert from "node:assert";
import { get_encoding } from "tiktoken";
import { chat } from "../llm/qwen.js";
import { config } from "../config.js";
import type { MemoryType } from "../db/queries.js";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ScoredCandidate {
  id: string;
  content: string;
  type: MemoryType;
  salience: number;
  confidence?: number;
  pinned: number;   // 0 | 1
  utility: number;  // Infinity for pinned
  session_id?: string | null;
}

export interface ManifestEntry {
  id: string;
  type: MemoryType;
  tokens: number;
  utility: number;
  confidence: number | null;
  session_born: string | null;
}

export interface PackedContext {
  text: string;
  manifest: ManifestEntry[];
  totalTokens: number;
}

// ─── Token counter (tiktoken o200k_base, process-lifetime singleton) ─────────

const _enc = get_encoding("o200k_base");

export function countTokens(text: string): number {
  return _enc.encode(text).length;
}

// ─── Internals ────────────────────────────────────────────────────────────────

const MIN_COMPRESS_TARGET = 20; // don't attempt compression below this target

function formatLine(m: ScoredCandidate): string {
  return `[${m.type}] ${m.content}`;
}

/** Deterministically truncate text to fit maxTokens (used only for pinned). */
function truncateToTokens(text: string, maxTokens: number): string {
  const tokens = _enc.encode(text);
  if (tokens.length <= maxTokens) return text;
  const sliced = _enc.decode(tokens.slice(0, Math.max(0, maxTokens - 1)));
  return `${new TextDecoder().decode(sliced)}…`;
}

async function compress(content: string, targetTokens: number): Promise<string> {
  const { text } = await chat({
    system:
      "You are a lossless memory compressor. Rewrite the input preserving all decision-relevant facts. Output only the rewritten text, no preamble.",
    messages: [
      {
        role: "user",
        content: `Rewrite in at most ${targetTokens} tokens, preserving all decision-relevant facts:\n\n${content}`,
      },
    ],
    model: config.ENGRAM_RERANK_MODEL, // qwen-flash: cheap, high-volume
    temperature: 0,
  });
  return text.trim();
}

// ─── packContext ──────────────────────────────────────────────────────────────

export async function packContext(
  candidates: ScoredCandidate[],
  budgetTokens: number,
): Promise<PackedContext> {
  const manifest: ManifestEntry[] = [];
  const lines: string[] = [];
  let usedTokens = 0;

  if (candidates.length === 0 || budgetTokens <= 0) {
    return { text: "", manifest, totalTokens: 0 };
  }

  const push = (m: ScoredCandidate, line: string, tokens: number) => {
    lines.push(line);
    manifest.push({
      id: m.id,
      type: m.type,
      tokens,
      utility: m.utility,
      confidence: m.confidence ?? null,
      session_born: m.session_id ?? null,
    });
    usedTokens += tokens;
  };

  /**
   * Try to fit a candidate into `remaining` tokens:
   * as-is → compressed → compressed retry (tighter) → truncated (pinned only).
   * Returns true if it was placed.
   */
  async function tryInsert(m: ScoredCandidate, remaining: number): Promise<boolean> {
    const line = formatLine(m);
    const lineTokens = countTokens(line);

    if (lineTokens <= remaining) {
      push(m, line, lineTokens);
      return true;
    }

    // Compression path — only when there's meaningful space to compress into.
    const headerTokens = countTokens(`[${m.type}] `);
    let target = remaining - headerTokens;
    if (target >= MIN_COMPRESS_TARGET) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const compressed = await compress(m.content, target);
          const compLine = `[${m.type}] ${compressed}`;
          const compTokens = countTokens(compLine);
          if (compTokens <= remaining) {
            push(m, compLine, compTokens);
            return true;
          }
          // Model overshot: retry once with a 20% tighter target.
          target = Math.floor(target * 0.8);
          if (target < MIN_COMPRESS_TARGET) break;
        } catch {
          break; // LLM failure — fall through
        }
      }
    }

    // Pinned guarantee vs hard budget: budget wins, so pinned gets truncated.
    if (m.pinned === 1 && remaining > headerTokens + 1) {
      const truncated = `[${m.type}] ${truncateToTokens(m.content, remaining - headerTokens)}`;
      const truncTokens = countTokens(truncated);
      if (truncTokens <= remaining) {
        push(m, truncated, truncTokens);
        return true;
      }
    }

    return false;
  }

  // ── Phase 1: reserved sub-budget for pinned + preference/decision ──────────
  const isCritical = (c: ScoredCandidate) =>
    c.pinned === 1 || c.type === "preference" || c.type === "decision";

  const byUtility = (a: ScoredCandidate, b: ScoredCandidate) => b.utility - a.utility;

  const critical = candidates.filter(isCritical).sort(byUtility);
  const reserveBudget = Math.floor(budgetTokens * config.ENGRAM_PACK_RESERVED_FRACTION);

  const leftovers: ScoredCandidate[] = [];
  for (const m of critical) {
    // Pinned may draw on the FULL budget (its guarantee outranks the reserve);
    // unpinned criticals are capped by the reserve in this phase.
    const remaining = m.pinned === 1
      ? budgetTokens - usedTokens
      : Math.min(reserveBudget, budgetTokens) - usedTokens;
    if (remaining <= 0 || !(await tryInsert(m, remaining))) {
      if (m.pinned !== 1) leftovers.push(m); // gets a second chance in phase 2
    }
  }

  // ── Phase 2: greedy fill by utility over the whole remaining budget ────────
  const rest = [...candidates.filter((c) => !isCritical(c)), ...leftovers].sort(byUtility);

  for (const m of rest) {
    const remaining = budgetTokens - usedTokens;
    if (remaining <= 0) break;
    await tryInsert(m, remaining);
  }

  // ── Hard guarantee (SPEC §4.3) ──────────────────────────────────────────────
  assert.ok(
    usedTokens <= budgetTokens,
    `packer invariant violated: ${usedTokens} > budget ${budgetTokens}`,
  );

  return {
    text: lines.join("\n"),
    manifest,
    totalTokens: usedTokens,
  };
}
