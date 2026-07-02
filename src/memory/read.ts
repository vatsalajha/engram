/**
 * src/memory/read.ts
 *
 * Read path: hybrid retrieval → RRF fusion → optional rerank → touch.
 *
 * hybridRetrieve — runs FTS5 (BM25) and sqlite-vec (cosine ANN) in parallel,
 *   fuses rankings with Reciprocal Rank Fusion (k=60), optionally reranks the
 *   shortlist with qwen-flash, then bumps access stats on returned memories.
 *
 * rerank — stand-alone export for callers that want to call it separately
 *   (e.g. the packer wants to rerank an already-retrieved set).
 *
 * Per SPEC §5.1:
 *   RRF score(m) = Σ 1 / (k + rank_i(m))   where k=60 (classic RRF constant)
 */

import { z } from "zod";
import { chatJSON, embed } from "../llm/qwen.js";
import { config } from "../config.js";
import {
  ftsSearch,
  vecSearch,
  touchMemory,
  type FtsResult,
  type VecResult,
  type MemoryType,
  type MemoryStatus,
} from "../db/queries.js";
import type { Db } from "../db/client.js";
import { packContext, type ScoredCandidate, type PackedContext } from "./packer.js";
import { utility } from "./scoring.js";

// ─── Result type ─────────────────────────────────────────────────────────────

export interface RetrievedMemory {
  id: string;
  content: string;
  type: MemoryType;
  status: MemoryStatus;
  salience: number;
  confidence: number;
  pinned: number;
  tags: string;
  created_at: number;
  session_id: string | null;
  // provenance
  rrfScore: number;
  ftsRank: number | null;   // 1-based position in the FTS leg (null if absent)
  vecRank: number | null;   // 1-based position in the vec leg (null if absent)
  rerankScore: number | null; // 0–10 from qwen-flash (null if not reranked)
}

// ─── RRF fusion ──────────────────────────────────────────────────────────────

interface RrfEntry {
  rrfScore: number;
  ftsRank: number | null;
  vecRank: number | null;
  data: Omit<RetrievedMemory, "rrfScore" | "ftsRank" | "vecRank" | "rerankScore" | "created_at"> & { created_at: number };
}

function rrfFuse(
  ftsResults: FtsResult[],
  vecResults: VecResult[],
  rrfK: number,
): RetrievedMemory[] {
  const entries = new Map<string, RrfEntry>();

  const upsert = (
    id: string,
    data: Omit<RetrievedMemory, "rrfScore" | "ftsRank" | "vecRank" | "rerankScore">,
    rank: number,
    leg: "fts" | "vec",
  ) => {
    const e = entries.get(id) ?? {
      rrfScore: 0,
      ftsRank: null,
      vecRank: null,
      data,
    };
    e.rrfScore += 1 / (rrfK + rank);
    if (leg === "fts") e.ftsRank = rank;
    else e.vecRank = rank;
    entries.set(id, e);
  };

  for (let i = 0; i < ftsResults.length; i++) {
    const r = ftsResults[i];
    upsert(r.id, { id: r.id, content: r.content, type: r.type, status: r.status,
      salience: r.salience, confidence: r.confidence, pinned: r.pinned, tags: r.tags,
      created_at: r.created_at, session_id: r.session_id }, i + 1, "fts");
  }

  for (let i = 0; i < vecResults.length; i++) {
    const r = vecResults[i];
    upsert(r.id, { id: r.id, content: r.content, type: r.type, status: r.status,
      salience: r.salience, confidence: r.confidence, pinned: r.pinned, tags: r.tags,
      created_at: r.created_at, session_id: r.session_id }, i + 1, "vec");
  }

  return Array.from(entries.values())
    .sort((a, b) => {
      if (b.rrfScore !== a.rrfScore) return b.rrfScore - a.rrfScore;
      // Tiebreak: prefer lower vec rank (semantically closer to query)
      const va = a.vecRank ?? Infinity;
      const vb = b.vecRank ?? Infinity;
      return va - vb;
    })
    .map((e) => ({
      ...e.data,
      rrfScore: e.rrfScore,
      ftsRank: e.ftsRank,
      vecRank: e.vecRank,
      rerankScore: null as number | null,
    }));
}

// ─── FTS query sanitiser ──────────────────────────────────────────────────────
// FTS5 MATCH rejects colons, brackets, and some operators; strip them.

function sanitiseFtsQuery(q: string): string {
  return q
    .replace(/[":*^()\[\]{}<>\\]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// ─── hybridRetrieve ───────────────────────────────────────────────────────────

export interface HybridRetrieveOptions {
  k?: number;                 // final top-k to return (default 24, SPEC §4.1)
  legK?: number;              // candidates per leg before fusion (overrides config; used by tests)
  queryEmbedding?: number[];  // skip embed() if already computed (also used by tests)
  skipRerank?: boolean;       // override config.ENGRAM_RERANK_ENABLED
}

/**
 * Hybrid retrieval for one user.
 *
 * 1. Embed the query (or use opts.queryEmbedding if provided).
 * 2. Run FTS5 and vec ANN in parallel (each leg fetches RETRIEVE_LEG_K candidates).
 * 3. Fuse with RRF.
 * 4. Optionally rerank the top RERANK_SHORTLIST with qwen-flash.
 * 5. Touch the returned memories (bump access stats).
 */
export async function hybridRetrieve(
  userId: string,
  query: string,
  db: Db,
  opts: HybridRetrieveOptions = {},
): Promise<RetrievedMemory[]> {
  const k        = opts.k ?? 24;
  const legK     = opts.legK ?? config.ENGRAM_RETRIEVE_LEG_K;
  const rrfK     = config.ENGRAM_RRF_K;
  const doRerank = (opts.skipRerank !== true) && config.ENGRAM_RERANK_ENABLED;

  // ── 1. Embed ──────────────────────────────────────────────────────────────
  const queryEmbedding: number[] = opts.queryEmbedding
    ?? (await embed([query]))[0];

  // ── 2. Parallel retrieval ─────────────────────────────────────────────────
  const safeFtsQuery = sanitiseFtsQuery(query);

  // FTS can throw if the sanitised query is empty or malformed; degrade gracefully.
  let ftsResults: FtsResult[] = [];
  if (safeFtsQuery.length > 0) {
    try {
      ftsResults = ftsSearch(db, userId, safeFtsQuery, legK);
    } catch {
      // FTS error (e.g. empty query after sanitisation) — fall back to vec-only
    }
  }

  const vecResults: VecResult[] = vecSearch(db, userId, queryEmbedding, legK);

  // ── 3. RRF fusion ─────────────────────────────────────────────────────────
  let fused = rrfFuse(ftsResults, vecResults, rrfK);

  // ── 4. Optionally rerank the shortlist ────────────────────────────────────
  if (doRerank && fused.length > 0) {
    const shortlist = fused.slice(0, config.ENGRAM_RERANK_SHORTLIST);
    const rest      = fused.slice(config.ENGRAM_RERANK_SHORTLIST);
    const reranked  = await rerank(query, shortlist);
    fused = [...reranked, ...rest];
  }

  // ── 5. Trim and touch ─────────────────────────────────────────────────────
  const final = fused.slice(0, k);
  for (const m of final) touchMemory(db, m.id);

  return final;
}

// ─── rerank ──────────────────────────────────────────────────────────────────

const RerankResponseSchema = z.object({
  scores: z.array(
    z.object({
      id: z.string(),
      score: z.number().min(0).max(10),
    })
  ),
});

/**
 * Ask qwen-flash to score each candidate's relevance to the query (0–10).
 * Returns candidates sorted best-first with rerankScore populated.
 * Falls back to the original RRF order if the API call fails.
 */
export async function rerank(
  query: string,
  candidates: RetrievedMemory[],
): Promise<RetrievedMemory[]> {
  if (candidates.length === 0) return candidates;

  const numbered = candidates
    .map((c, i) => `[${i + 1}] id=${c.id}\n${c.content}`)
    .join("\n\n");

  const systemPrompt = `\
You are a relevance scoring engine. Given a user query and a numbered list of memory items, score each item's relevance to answering or informing that query.

Score 0–10:
  10 = directly answers or is essential context
   7 = clearly relevant, useful background
   4 = loosely related
   0 = irrelevant

Return ONLY valid JSON (no prose, no fences):
{"scores":[{"id":"<id>","score":<0-10>},...]}

One entry per memory item. Include every id exactly once.`;

  try {
    const result = await chatJSON(
      {
        system: systemPrompt,
        messages: [
          {
            role: "user",
            content: `Query: "${query}"\n\nMemories:\n${numbered}`,
          },
        ],
        model: config.ENGRAM_RERANK_MODEL,
        temperature: 0,
      },
      RerankResponseSchema,
    );

    // Build score lookup
    const scoreMap = new Map(result.scores.map((s) => [s.id, s.score]));

    return candidates
      .map((c) => ({ ...c, rerankScore: scoreMap.get(c.id) ?? null }))
      .sort((a, b) => {
        // Null scores fall to the bottom
        const sa = a.rerankScore ?? -1;
        const sb = b.rerankScore ?? -1;
        return sb - sa;
      });
  } catch {
    // Rerank failure is non-fatal; return original RRF order
    return candidates;
  }
}

// ─── recall ───────────────────────────────────────────────────────────────────

export interface RecallDetail {
  packed: PackedContext;
  /** The ranked retrieval candidates BEFORE packing (for recall@k metrics). */
  retrieved: RetrievedMemory[];
}

/**
 * Full recall pipeline: retrieve → rerank → score → pack.
 * Returns a packed context string + manifest ready for injection into a prompt.
 *
 * @param budget  Token budget (defaults to config.ENGRAM_CONTEXT_BUDGET).
 */
export async function recall(
  userId: string,
  query: string,
  db: Db,
  budget?: number,
): Promise<PackedContext> {
  return (await recallDetailed(userId, query, db, budget)).packed;
}

/** recall() plus the pre-packing candidate list — used by the eval harness. */
export async function recallDetailed(
  userId: string,
  query: string,
  db: Db,
  budget?: number,
): Promise<RecallDetail> {
  const resolvedBudget = budget ?? config.ENGRAM_CONTEXT_BUDGET;

  // hybridRetrieve handles embed + FTS + vec + RRF + optional rerank
  const retrieved = await hybridRetrieve(userId, query, db);

  const now = Date.now();
  const scored: ScoredCandidate[] = retrieved.map((m) => {
    const ageDays = (now - m.created_at) / 86_400_000;
    // Normalise relevance: prefer rerankScore (0–10 → 0–1); fall back to RRF normalisation
    const relevance =
      m.rerankScore !== null
        ? m.rerankScore / 10
        : Math.min(1, m.rrfScore * (config.ENGRAM_RRF_K + 1) / 2);
    const u = utility(
      { type: m.type, salience: m.salience, confidence: m.confidence, pinned: m.pinned },
      relevance,
      ageDays,
    );
    return {
      id: m.id,
      content: m.content,
      type: m.type,
      salience: m.salience,
      confidence: m.confidence,
      pinned: m.pinned,
      utility: u,
      session_id: m.session_id,
    };
  });

  const packed = await packContext(scored, resolvedBudget);
  return { packed, retrieved };
}
