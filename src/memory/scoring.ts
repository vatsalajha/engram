/**
 * src/memory/scoring.ts
 *
 * Decision-utility score (for context packing) and retention score (for forgetting).
 *
 * utility  = w_rel·relevance + w_sal·salience + w_rec·recency + w_typ·typePriority
 * retention = w_sal·sal + w_con·conf + w_rec·recency + w_acc·logAccess + w_typ·typePrio
 *
 * recency(ageDays) = 1 / (1 + ageDays * λ)   — exponential-style decay
 */

import { config } from "../config.js";
import type { MemoryType } from "../db/queries.js";

// ─── Type priority map ────────────────────────────────────────────────────────

export const TYPE_PRIORITY: Record<MemoryType, number> = {
  preference: 1.00,
  decision:   0.85,
  fact:       0.70,
  skill:      0.55,
  hypothesis: 0.45,   // must earn confidence before outranking established types
  event:      0.35,
  note:       0.20,
};

// ─── Recency ──────────────────────────────────────────────────────────────────

export function recency(ageDays: number): number {
  return 1 / (1 + ageDays * config.ENGRAM_RECENCY_LAMBDA);
}

// ─── Decision-utility score ───────────────────────────────────────────────────

/**
 * How useful a memory is right now for answering the current query (SPEC §4.2).
 * Confidence is a first-class term in v2: memories that earned belief through
 * confirmed outcomes outrank equally-relevant unproven ones.
 *
 * Pinned memories return Infinity — always included if they fit.
 *
 * @param memory    The memory being scored.
 * @param relevance Normalised 0–1 relevance to the query
 *                  (rerankScore/10, or normalised RRF score).
 * @param ageDays   How many days since the memory was created.
 */
export function utility(
  memory: { type: MemoryType; salience: number; confidence: number; pinned?: number },
  relevance: number,
  ageDays: number,
): number {
  if (memory.pinned === 1) return Infinity;

  const rel = Math.min(1, Math.max(0, relevance));
  const sal = Math.min(1, Math.max(0, memory.salience));
  const con = Math.min(1, Math.max(0, memory.confidence));
  const rec = recency(ageDays);
  const typ = TYPE_PRIORITY[memory.type] ?? 0.2;

  return (
    config.ENGRAM_SCORE_W_REL * rel +
    config.ENGRAM_SCORE_W_SAL * sal +
    config.ENGRAM_SCORE_W_CON * con +
    config.ENGRAM_SCORE_W_REC * rec +
    config.ENGRAM_SCORE_W_TYP * typ
  );
}

// ─── Retention score (forgetting engine) ─────────────────────────────────────

/**
 * How worth keeping a memory is over time (SPEC §4.5, 4 terms in v2).
 * Low score → candidate for archival/expiry.
 *
 * logAccess is normalised by log(101) so 100 accesses ≈ 1.0.
 */
export function retention(memory: {
  salience: number;
  confidence: number;
  access_count: number;
  created_at: number;
}): number {
  const ageDays = (Date.now() - memory.created_at) / 86_400_000;
  const rec = recency(ageDays);
  const logAccess = Math.log1p(memory.access_count) / Math.log1p(100);

  return (
    config.ENGRAM_RETENTION_W_SAL * Math.min(1, memory.salience) +
    config.ENGRAM_RETENTION_W_CON * Math.min(1, memory.confidence) +
    config.ENGRAM_RETENTION_W_REC * rec +
    config.ENGRAM_RETENTION_W_ACC * Math.min(1, logAccess)
  );
}
