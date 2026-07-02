/**
 * src/memory/belief.ts
 *
 * Belief engine (SPEC §4.4) — the mechanism behind "increasingly accurate
 * decisions": memories earn or lose confidence based on decision outcomes.
 *
 * applyFeedback  — explicit signal (POST /feedback, the eval harness, or the
 *                  implicit judge). confidence ← clamp01(conf ± η); every
 *                  change is logged to belief_events; floor enforcement:
 *                  hypotheses auto-archive, facts/preferences → needs_review.
 * implicitJudge  — during reflect: qwen-plus judges whether the user's next
 *                  turn supports/contradicts any memory used in the packed
 *                  context (manifest). Non-neutral verdicts feed applyFeedback.
 *
 * Invariants: all updates transactional; belief_events is append-only;
 * η and the floor live in src/config.ts (no magic numbers).
 */

import { z } from "zod";
import { chatJSON } from "../llm/qwen.js";
import { config } from "../config.js";
import type { Db } from "../db/client.js";
import {
  getMemory,
  setConfidence,
  setNeedsReview,
  updateMemoryStatus,
  addEdge,
  insertBeliefEvent,
  type Memory,
} from "../db/queries.js";

// ─── Types ────────────────────────────────────────────────────────────────────

export type FeedbackOutcome = "confirmed" | "contradicted";

export interface FeedbackResult {
  memory_id: string;
  outcome: FeedbackOutcome;
  confidence_before: number;
  confidence_after: number;
  floor_action: "none" | "archived" | "needs_review";
}

export interface ApplyFeedbackOptions {
  /**
   * Optional memory that anchors the graph edge (e.g. a memory extracted from
   * the turn that produced this feedback). When present, a supports/contradicts
   * edge anchor → memory is added. belief_events carries turn provenance
   * regardless, so feedback without an anchor is still fully audited.
   */
  anchorMemoryId?: string | null;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

export function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

// ─── applyFeedback ────────────────────────────────────────────────────────────

/**
 * Apply a confirmed/contradicted outcome to a set of memories.
 * Per memory (single transaction):
 *   1. confidence ← clamp01(confidence + η·(+1|−1))
 *   2. belief_events row (delta = actual applied change, turn provenance)
 *   3. supports/contradicts edge from the anchor memory, if one is given
 *   4. floor enforcement when confidence < ENGRAM_CONFIDENCE_FLOOR:
 *        hypothesis           → status = archived
 *        fact/preference/etc. → needs_review = 1
 */
export function applyFeedback(
  db: Db,
  userId: string,
  turnId: string | null,
  outcome: FeedbackOutcome,
  memoryIds: string[],
  opts: ApplyFeedbackOptions = {},
): FeedbackResult[] {
  const eta = config.ENGRAM_BELIEF_ETA;
  const floor = config.ENGRAM_CONFIDENCE_FLOOR;
  const sign = outcome === "confirmed" ? 1 : -1;
  const results: FeedbackResult[] = [];

  const run = db.transaction(() => {
    for (const id of memoryIds) {
      const mem = getMemory(db, id);
      // Scope guard: never mutate another user's memories; skip missing/inactive.
      if (!mem || mem.user_id !== userId || mem.status !== "active") continue;

      const before = mem.confidence;
      const after = clamp01(before + eta * sign);
      setConfidence(db, id, after);

      insertBeliefEvent(db, {
        memory_id: id,
        delta: after - before, // actual applied change (0 at the clamp rails)
        reason: outcome,
        turn_id: turnId,
      });

      if (opts.anchorMemoryId && opts.anchorMemoryId !== id) {
        addEdge(
          db,
          opts.anchorMemoryId,
          id,
          outcome === "confirmed" ? "supports" : "contradicts",
        );
      }

      let floorAction: FeedbackResult["floor_action"] = "none";
      if (after < floor) {
        if (mem.type === "hypothesis") {
          // Hypotheses must earn their keep — below the floor they die.
          updateMemoryStatus(db, id, "archived");
          floorAction = "archived";
        } else {
          setNeedsReview(db, id, true);
          floorAction = "needs_review";
        }
      }

      results.push({
        memory_id: id,
        outcome,
        confidence_before: before,
        confidence_after: after,
        floor_action: floorAction,
      });
    }
  });

  run();
  return results;
}

// ─── implicitJudge ────────────────────────────────────────────────────────────

const JudgeSchema = z.object({
  verdicts: z.array(
    z.object({
      index: z.number().int().min(1),
      verdict: z.enum(["supported", "contradicted", "neutral"]),
      reasoning: z.string(),
    }),
  ),
});

export interface ImplicitJudgeResult {
  judged: number;                 // memories evaluated
  applied: FeedbackResult[];      // non-neutral outcomes applied
}

/**
 * During reflect: given the memories that were packed into the turn's context
 * (the manifest) and the turn itself, ask qwen-plus which memories the turn's
 * outcome SUPPORTED or CONTRADICTED. Neutral verdicts are ignored.
 *
 * One LLM call for the whole manifest. Failures are non-fatal and return
 * { judged: 0, applied: [] } — belief updates are best-effort by design.
 */
export async function implicitJudge(
  db: Db,
  userId: string,
  turnId: string | null,
  manifest: Array<{ id: string }>,
  turn: { user: string; assistant?: string },
  opts: ApplyFeedbackOptions = {},
): Promise<ImplicitJudgeResult> {
  // Resolve manifest ids to live memories (scope-guarded).
  const memories = manifest
    .map((m) => getMemory(db, m.id))
    .filter((m): m is Memory => !!m && m.user_id === userId && m.status === "active");
  if (memories.length === 0) return { judged: 0, applied: [] };

  const numbered = memories.map((m, i) => `${i + 1}. ${m.content}`).join("\n");
  const turnText = turn.assistant
    ? `[USER TURN]\n${turn.user}\n\n[ASSISTANT RESPONSE]\n${turn.assistant}`
    : `[USER TURN]\n${turn.user}`;

  let judged: z.infer<typeof JudgeSchema>;
  try {
    judged = await chatJSON(
      {
        system: `You audit an agent's memory. Given a numbered list of MEMORIES that were used to answer a turn, and the TURN itself, judge each memory:
  supported     — the turn's content confirms the memory is (still) true
  contradicted  — the turn's content conflicts with the memory
  neutral       — the turn says nothing about this memory
Judge ONLY from what the turn actually says; when in doubt, return neutral.
Return JSON only: {"verdicts":[{"index":1,"verdict":"supported"|"contradicted"|"neutral","reasoning":"..."}, …]}
Include exactly one verdict per memory.`,
        messages: [{ role: "user", content: `MEMORIES:\n${numbered}\n\n${turnText}` }],
        model: config.ENGRAM_EXTRACT_MODEL,
        temperature: 0,
      },
      JudgeSchema,
    );
  } catch {
    return { judged: 0, applied: [] };
  }

  const confirmed: string[] = [];
  const contradicted: string[] = [];
  for (const v of judged.verdicts) {
    const mem = memories[v.index - 1];
    if (!mem) continue; // hallucinated index
    if (v.verdict === "supported") confirmed.push(mem.id);
    else if (v.verdict === "contradicted") contradicted.push(mem.id);
  }

  const applied = [
    ...applyFeedback(db, userId, turnId, "confirmed", confirmed, opts),
    ...applyFeedback(db, userId, turnId, "contradicted", contradicted, opts),
  ];

  return { judged: memories.length, applied };
}
