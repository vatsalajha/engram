/**
 * src/memory/sleep.ts (formerly maintenance.ts)
 *
 * The sleep cycle (SPEC §4.5): what the agent does between sessions.
 *
 * detectSupersession — write-time forgetting; called from writeMemories.
 * consolidate        — episodic→semantic: clusters events, summarises, edges.
 * decayAndExpire     — retention-scored sweep; archives stale, expires dead.
 * inferHypotheses    — "the agent dreams": proposes low-confidence hypotheses
 *                      from the episodic log; they earn belief or die.
 * runSleep           — orchestrates all of it, per-user locked, audit-logged.
 */

import { z } from "zod";
import { chat, chatJSON, embed } from "../llm/qwen.js";
import { config } from "../config.js";
import type { Db } from "../db/client.js";
import {
  listMemories,
  insertMemory,
  updateMemoryStatus,
  addEdge,
  insertBeliefEvent,
  insertSleepRun,
  recentEpisodic,
  vecSearch,
  type Memory,
  type MemoryType,
} from "../db/queries.js";
import { retention } from "./scoring.js";

// ─── detectSupersession ───────────────────────────────────────────────────────

const SupersessionBatchSchema = z.object({
  verdicts: z.array(
    z.object({
      old_index: z.number().int().min(1),
      action: z.enum(["update", "contradict", "complement", "unrelated"]),
      reasoning: z.string(),
    }),
  ),
});

/**
 * Given a newly written memory, check whether it supersedes any existing ones.
 * Compares against the top-5 nearest active memories of related types with a
 * SINGLE qwen-plus call (one verdict per pair).
 *
 * On update/contradict: old → status='superseded' + superseded_by=new.id,
 * a `supersedes` edge (new → old) in the memory graph, and a belief_event
 * (reason='contradicted') on the old memory — the audit trail of forgetting.
 *
 * Idempotent: safe to call multiple times for the same new memory.
 */
export async function detectSupersession(
  userId: string,
  newMemory: {
    id: string;
    type: MemoryType;
    content: string;
    embedding: number[];
  },
  db: Db,
  turnId?: string | null,
): Promise<void> {
  // Find nearest existing active memories (exclude the new one itself)
  const candidates = vecSearch(db, userId, newMemory.embedding, 5).filter(
    (c) => c.id !== newMemory.id && c.status === "active",
  );

  // Only check semantically related types for supersession
  const FACT_LIKE = new Set<MemoryType>(["preference", "decision", "fact"]);
  const isSameCluster =
    FACT_LIKE.has(newMemory.type)
      ? (t: MemoryType) => FACT_LIKE.has(t)
      : (t: MemoryType) => t === newMemory.type;

  const pairs = candidates.filter((c) => isSameCluster(c.type));
  if (pairs.length === 0) return;

  const oldList = pairs
    .map((c, i) => `${i + 1}. ${c.content}`)
    .join("\n");

  let result: z.infer<typeof SupersessionBatchSchema>;
  try {
    result = await chatJSON(
      {
        system: `You compare a NEW memory statement against a numbered list of OLD ones and return one verdict per OLD item.
Return JSON only: {"verdicts":[{"old_index":1,"action":"update"|"contradict"|"complement"|"unrelated","reasoning":"..."}, …]}
  update     — NEW directly replaces or updates OLD (e.g. changed preference, corrected fact)
  contradict — NEW conflicts with OLD but doesn't clearly replace it
  complement — both are compatible and add distinct information
  unrelated  — no meaningful relationship
Include exactly one verdict for every OLD item.`,
        messages: [
          {
            role: "user",
            content: `OLD memories:\n${oldList}\n\nNEW: ${newMemory.content}`,
          },
        ],
        model: config.ENGRAM_EXTRACT_MODEL,
        temperature: 0,
      },
      SupersessionBatchSchema,
    );
  } catch {
    return; // LLM error is non-fatal; the write already succeeded
  }

  const supersede = db.transaction((old: { id: string }, reasoning: string) => {
    updateMemoryStatus(db, old.id, "superseded", newMemory.id);
    addEdge(db, newMemory.id, old.id, "supersedes");
    insertBeliefEvent(db, {
      memory_id: old.id,
      delta: -config.ENGRAM_BELIEF_ETA,
      reason: "contradicted",
      turn_id: turnId ?? null,
    });
  });

  for (const verdict of result.verdicts) {
    if (verdict.action !== "update" && verdict.action !== "contradict") continue;
    const old = pairs[verdict.old_index - 1];
    if (!old) continue;              // model hallucinated an index — ignore
    if (old.status !== "active") continue; // double-supersession guard
    supersede(old, verdict.reasoning);
  }
}

// ─── decayAndExpire ───────────────────────────────────────────────────────────

export interface DecayResult {
  archived: number;
  expired: number;
}

/**
 * Compute retention scores for all active, non-pinned memories of a user.
 * Archive those below ENGRAM_ARCHIVE_THRESHOLD.
 * Expire any archived memory whose ttl_at is in the past.
 * Transactional per batch; never touches pinned memories.
 */
export function decayAndExpire(userId: string, db: Db): DecayResult {
  const now = Date.now();

  // 1. Active, non-pinned memories only
  const active = listMemories(db, {
    user_id: userId,
    status: "active",
    pinned: false,
    limit: 5000,
  });

  // 2. Classify by retention score
  const toArchive = active
    .filter((m) => retention(m) < config.ENGRAM_ARCHIVE_THRESHOLD)
    .map((m) => m.id);

  // 3. Batch-archive (single transaction)
  if (toArchive.length > 0) {
    db.transaction(() => {
      for (const id of toArchive) {
        db.prepare("UPDATE memories SET status = 'archived' WHERE id = ?").run(id);
      }
    })();
  }

  // 4. Expire archived memories: past their TTL, or archived-and-untouched
  //    for ENGRAM_EXPIRE_ARCHIVED_HOURS (48h default).
  const staleCutoff = now - config.ENGRAM_EXPIRE_ARCHIVED_HOURS * 3_600_000;
  const toExpire = (
    db
      .prepare(
        `SELECT id FROM memories
         WHERE user_id = ? AND status = 'archived' AND pinned = 0
           AND ((ttl_at IS NOT NULL AND ttl_at < ?) OR last_accessed_at < ?)`,
      )
      .all(userId, now, staleCutoff) as { id: string }[]
  ).map((r) => r.id);

  if (toExpire.length > 0) {
    db.transaction(() => {
      for (const id of toExpire) {
        db.prepare("UPDATE memories SET status = 'expired' WHERE id = ?").run(id);
      }
    })();
  }

  return { archived: toArchive.length, expired: toExpire.length };
}

// ─── consolidate ──────────────────────────────────────────────────────────────

/**
 * Cluster active event memories by ANN similarity (pairwise-to-seed cosine
 * ≥ ENGRAM_CONSOLIDATE_SIM). For each cluster of size ≥ ENGRAM_CONSOLIDATE_MIN:
 *   - summarise with qwen-plus into one source='consolidated' memory
 *   - embed the summary so it participates in future vector recall
 *   - link summary → each source with a derived_from edge (the memory graph)
 *   - archive the source events
 *
 * Returns the number of consolidations created.
 */
export async function consolidate(userId: string, db: Db): Promise<number> {
  // Fetch all active event memories that have embeddings
  const allEvents = listMemories(db, {
    user_id: userId,
    status: "active",
    type: "event",
    limit: 5000,
  }).filter((e): e is Memory & { embedding: number[] } =>
    Array.isArray(e.embedding) && e.embedding.length > 0,
  );

  if (allEvents.length < config.ENGRAM_CONSOLIDATE_MIN) return 0;

  // Greedy clustering: seed → ANN (similarity-gated) → cluster
  const maxDistance = 1 - config.ENGRAM_CONSOLIDATE_SIM; // cosine distance gate
  const visited = new Set<string>();
  const clusters: Array<Memory & { embedding: number[] }>[] = [];

  for (const seed of allEvents) {
    if (visited.has(seed.id)) continue;

    const neighbors = vecSearch(db, userId, seed.embedding, config.ENGRAM_CONSOLIDATE_K)
      .filter(
        (n) => n.type === "event" && !visited.has(n.id) && n.distance <= maxDistance,
      );

    // Map neighbor ids back to full Memory objects (we need embeddings for re-seeding)
    const cluster = neighbors
      .map((n) => allEvents.find((e) => e.id === n.id))
      .filter((e): e is Memory & { embedding: number[] } => e != null);

    if (cluster.length >= config.ENGRAM_CONSOLIDATE_MIN) {
      cluster.forEach((m) => visited.add(m.id));
      clusters.push(cluster);
    } else {
      visited.add(seed.id);
    }
  }

  let count = 0;

  for (const cluster of clusters) {
    const eventTexts = cluster
      .map((m, i) => `${i + 1}. ${m.content}`)
      .join("\n");

    let summary: string;
    try {
      const { text } = await chat({
        system:
          "Summarize the following related events into a single, self-contained semantic memory statement. Be concise, third-person, and preserve all key facts. Output only the summary sentence, no preamble.",
        messages: [{ role: "user", content: eventTexts }],
        model: config.ENGRAM_EXTRACT_MODEL,
        temperature: 0.3,
      });
      summary = text.trim();
    } catch {
      continue; // non-fatal; skip this cluster
    }

    // Embed the summary (best-effort — without it the memory still exists,
    // it just won't join vector recall until re-embedded).
    let summaryEmbedding: number[] | undefined;
    try {
      [summaryEmbedding] = await embed([summary]);
    } catch {
      summaryEmbedding = undefined;
    }

    // Store summary + archive sources + graph edges in ONE transaction.
    db.transaction(() => {
      const summaryId = insertMemory(db, {
        user_id: userId,
        type: "fact",
        content: summary,
        embedding: summaryEmbedding,
        salience: Math.max(...cluster.map((m) => m.salience)),
        confidence: 0.8,
        source: "consolidated",
      });
      for (const m of cluster) {
        addEdge(db, summaryId, m.id, "derived_from");
        db.prepare("UPDATE memories SET status = 'archived' WHERE id = ?").run(m.id);
      }
    })();

    count++;
  }

  return count;
}

// ─── inferHypotheses ("the agent dreams") ─────────────────────────────────────

const InferSchema = z.object({
  hypotheses: z.array(
    z.object({
      content: z.string().min(10),
      reasoning: z.string(),
    }),
  ),
});

/**
 * Feed the recent episodic log + the user's top preferences to qwen-plus and
 * ask for up to ENGRAM_INFER_MAX NEW hypotheses about the user — patterns the
 * user never stated but the transcript suggests.
 *
 * Newborn hypotheses enter at ENGRAM_HYPOTHESIS_CONFIDENCE (0.4) with
 * source='inferred' and must earn confidence via the belief engine or die
 * in decay. Proposals semantically close to an existing active memory
 * (cosine ≥ ENGRAM_INFER_DEDUPE_SIM) are dropped.
 *
 * Returns ids of inserted hypotheses.
 */
export async function inferHypotheses(userId: string, db: Db): Promise<string[]> {
  const episodic = recentEpisodic(db, userId, config.ENGRAM_INFER_EPISODIC_N);
  if (episodic.length === 0) return []; // nothing observed → nothing to dream about

  const transcript = [...episodic]
    .reverse() // chronological
    .map((e) => `${e.role}: ${e.content}`)
    .join("\n");

  const existing = [
    ...listMemories(db, { user_id: userId, status: "active", type: "preference", limit: 10 }),
    ...listMemories(db, { user_id: userId, status: "active", type: "hypothesis", limit: 10 }),
  ];
  const existingList = existing.length
    ? existing.map((m) => `- ${m.content}`).join("\n")
    : "(none)";

  let proposed: z.infer<typeof InferSchema>;
  try {
    proposed = await chatJSON(
      {
        system: `You are Engram's inference engine, run while the agent "sleeps". Given a raw conversation transcript and the memories already stored, propose up to ${config.ENGRAM_INFER_MAX} NEW hypotheses about this user.

RULES:
1. A hypothesis is a PATTERN the user never stated outright but the transcript suggests (habits, tendencies, working style, tastes).
2. Do NOT restate anything semantically present in EXISTING MEMORIES.
3. Each hypothesis must be atomic, third-person, self-contained: "The user tends to …".
4. Only propose what the transcript actually supports. Fewer, better hypotheses beat filler — return an empty list if nothing genuine emerges.

Return JSON only: {"hypotheses":[{"content":"...","reasoning":"..."}]}`,
        messages: [
          {
            role: "user",
            content: `EXISTING MEMORIES:\n${existingList}\n\nTRANSCRIPT:\n${transcript}`,
          },
        ],
        model: config.ENGRAM_EXTRACT_MODEL,
        temperature: 0.4, // some creativity — these are guesses by design
      },
      InferSchema,
    );
  } catch {
    return []; // inference is best-effort; sleep continues
  }

  const candidates = proposed.hypotheses.slice(0, config.ENGRAM_INFER_MAX);
  if (candidates.length === 0) return [];

  // Embed proposals for dedupe + storage.
  let embeddings: number[][];
  try {
    embeddings = await embed(candidates.map((h) => h.content));
  } catch {
    return [];
  }

  const inserted: string[] = [];
  const maxDistance = 1 - config.ENGRAM_INFER_DEDUPE_SIM;

  for (let i = 0; i < candidates.length; i++) {
    const h = candidates[i];
    const nearest = vecSearch(db, userId, embeddings[i], 1)[0];
    if (nearest && nearest.distance <= maxDistance) continue; // already known

    inserted.push(
      insertMemory(db, {
        user_id: userId,
        type: "hypothesis",
        content: h.content,
        embedding: embeddings[i],
        salience: 0.5,
        confidence: config.ENGRAM_HYPOTHESIS_CONFIDENCE,
        source: "inferred",
        tags: ["inferred"],
      }),
    );
  }

  return inserted;
}

// ─── runSleep ─────────────────────────────────────────────────────────────────

export interface SleepRunResult {
  id: string | null;       // sleep_runs row id (null when skipped)
  user_id: string;
  consolidated_n: number;
  decayed_n: number;       // archived + expired
  inferred_n: number;
  notes: string;
  skipped: boolean;        // another run for this user is already in flight
}

// In-process per-user mutex. Distributed locking (Redis SET NX) lives one
// level up in the scheduler's LockStore; this guards direct POST /sleep calls
// racing the scheduler inside a single process.
const _sleeping = new Set<string>();

/**
 * One full sleep cycle for a user: consolidate → decay → infer → audit row.
 * Idempotent and safe to run repeatedly: a second immediate run finds no
 * clusters, nothing newly stale, and dedupes re-proposed hypotheses.
 */
export async function runSleep(userId: string, db: Db): Promise<SleepRunResult> {
  if (_sleeping.has(userId)) {
    return {
      id: null, user_id: userId,
      consolidated_n: 0, decayed_n: 0, inferred_n: 0,
      notes: "skipped: sleep already in progress", skipped: true,
    };
  }

  _sleeping.add(userId);
  try {
    const consolidated_n = await consolidate(userId, db);
    const { archived, expired } = decayAndExpire(userId, db);
    const inferredIds = await inferHypotheses(userId, db);

    const notes = JSON.stringify({ archived, expired, inferred_ids: inferredIds });
    const id = insertSleepRun(db, {
      user_id: userId,
      consolidated_n,
      decayed_n: archived + expired,
      inferred_n: inferredIds.length,
      notes,
    });

    return {
      id, user_id: userId,
      consolidated_n, decayed_n: archived + expired, inferred_n: inferredIds.length,
      notes, skipped: false,
    };
  } finally {
    _sleeping.delete(userId);
  }
}

