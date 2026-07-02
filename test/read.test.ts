/**
 * test/read.test.ts
 *
 * PART A — offline (no API): RRF correctness, hybrid beats BM25-only, stability.
 * PART B — integration (DASHSCOPE_API_KEY required): live embed + optional rerank.
 *
 * Key insight: in a tiny in-memory store, vecSearch(k=20) returns EVERY memory
 * (even orthogonal ones). Tests that need "decoy is vec-only-absent" use
 * legK:1 or legK:2 to pin exactly which documents each leg fetches.
 *
 * Run: npm test
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openDb, type Db } from "../src/db/client.js";
import { insertMemory, ftsSearch, vecSearch, getMemory } from "../src/db/queries.js";
import { hybridRetrieve, rerank } from "../src/memory/read.js";

// ─── Synthetic embedding helpers ─────────────────────────────────────────────

/** Vector with cosine similarity `sim` to unitVec(d) (mixed with dim d+1). */
function nearVec(d: number, sim: number, dim = 1024): number[] {
  const perp = Math.sqrt(1 - sim * sim);
  const v = new Array<number>(dim).fill(0);
  v[d] = sim;
  v[d + 1] = perp;
  return v;
}

function unitVec(d: number, dim = 1024): number[] {
  const v = new Array<number>(dim).fill(0);
  v[d] = 1.0;
  return v;
}

// ─── PART A: offline tests ────────────────────────────────────────────────────

describe("hybridRetrieve (offline)", () => {
  let db: Db;

  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  /**
   * Scenario (legK=1 — each leg returns exactly its top-1 result):
   *
   *   gold   embedding=unitVec(0)  content="The user rises before dawn each day."
   *          → vec rank 1  /  FTS: no keyword overlap with query "morning routine habits"
   *
   *   decoy  embedding=unitVec(100)  content="Morning routine habits define productivity."
   *          → fts rank 1  /  vec rank 2 (orthogonal, but that's outside legK=1)
   *
   * With legK=1: vec leg = [gold], fts leg = [decoy].
   * RRF: gold = 1/(60+1), decoy = 1/(60+1). Tie on RRF score — but both surface.
   */

  let goldId: string;
  let decoyId: string;
  let noiseId: string;

  before(() => {
    goldId = insertMemory(db, {
      user_id: "u_read",
      type: "preference",
      content: "The user rises before dawn each day and exercises immediately.",
      salience: 0.8,
      embedding: unitVec(0),
    });

    decoyId = insertMemory(db, {
      user_id: "u_read",
      type: "fact",
      content: "Morning routine habits and daily schedules help the user stay productive.",
      salience: 0.5,
      embedding: unitVec(100),
    });

    noiseId = insertMemory(db, {
      user_id: "u_read",
      type: "note",
      content: "The user enjoys cooking pasta on weekends.",
      salience: 0.3,
      embedding: unitVec(200),
    });
  });

  it("pure BM25 (ftsSearch alone) misses the gold memory", () => {
    const results = ftsSearch(db, "u_read", "morning routine habits", 10);
    const ids = results.map((r) => r.id);
    assert.ok(!ids.includes(goldId), "pure BM25 should NOT find the gold (no keyword overlap)");
    assert.ok(ids.includes(decoyId), "pure BM25 SHOULD find the decoy (keyword match)");
  });

  it("pure vecSearch returns gold at rank 1 and ranks decoy below gold", () => {
    const results = vecSearch(db, "u_read", unitVec(0), 10);
    assert.ok(results.length > 0, "vecSearch returned nothing");
    assert.equal(results[0].id, goldId, "gold should be vec rank 1");
    const decoyPos = results.findIndex((r) => r.id === decoyId);
    assert.ok(decoyPos > 0, "decoy should be below gold in vec (orthogonal embedding)");
  });

  it("hybrid (legK=1) surfaces the semantic-only gold that BM25 alone misses", async () => {
    // legK=1: vec leg = [gold], fts leg = [decoy]. Both appear after RRF.
    const results = await hybridRetrieve("u_read", "morning routine habits", db, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
      legK: 1,
      k: 5,
    });

    const ids = results.map((r) => r.id);
    assert.ok(ids.includes(goldId), "hybrid should surface the semantic-only gold");
    assert.ok(ids.includes(decoyId), "hybrid should surface the lexical-only decoy");
  });

  it("hybrid (legK=1) excludes noise that appears in neither leg", async () => {
    const results = await hybridRetrieve("u_read", "morning routine habits", db, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
      legK: 1,
      k: 5,
    });
    const ids = results.map((r) => r.id);
    assert.ok(!ids.includes(noiseId), "noise should not appear — it was in neither leg");
  });

  /**
   * RRF ranking test (legK=1):
   *   vec leg: [gold rank1]    — gold is exact embedding match, decoy is outside legK=1
   *   fts leg: [decoy rank1]   — decoy has the query keywords, gold has none
   *
   *   RRF scores: gold = 1/(60+1), decoy = 1/(60+1)  → tie
   *
   *   Tiebreaker: vecRank (gold=1) < vecRank (decoy=null→∞)
   *   → gold wins and appears before decoy in the ranked list.
   *
   * This proves that when scores are equal, semantic proximity beats keyword-only.
   */
  it("RRF ranking: vec-tiebreaker places semantic gold above equal-score lexical decoy", async () => {
    const db2 = openDb(":memory:");

    const g = insertMemory(db2, {
      user_id: "u_rank",
      type: "preference",
      content: "The user prefers standing desks while coding.",   // zero keyword overlap with query
      salience: 0.8,
      embedding: unitVec(0),   // exact query embedding
    });

    const d = insertMemory(db2, {
      user_id: "u_rank",
      type: "fact",
      content: "Morning routine habits define a productive workday.",  // keyword match
      salience: 0.4,
      embedding: unitVec(100),   // orthogonal — outside vec legK=1
    });

    // legK=1: vec returns only gold; fts returns only decoy.
    // Both get RRF = 1/(60+1) ≈ 0.01639 — exact tie.
    // Tiebreaker fires: gold.vecRank(1) < decoy.vecRank(null→∞) → gold is first.
    const results = await hybridRetrieve("u_rank", "morning routine habits", db2, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
      legK: 1,
      k: 5,
    });

    const ids = results.map((r) => r.id);
    assert.ok(ids.includes(g), "gold must be in hybrid results");
    assert.ok(ids.includes(d), "decoy must be in hybrid results");

    const gIdx = ids.indexOf(g);
    const dIdx = ids.indexOf(d);
    assert.ok(gIdx < dIdx, `gold (idx ${gIdx}) should rank above decoy (idx ${dIdx}) via vec tiebreaker`);

    db2.close();
  });

  it("RRF provenance: ftsRank and vecRank correctly populated (legK=1)", async () => {
    // legK=1: vec returns only gold (rank 1), fts returns only decoy (rank 1).
    const results = await hybridRetrieve("u_read", "morning routine habits", db, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
      legK: 1,
      k: 5,
    });

    const gold  = results.find((r) => r.id === goldId);
    const decoy = results.find((r) => r.id === decoyId);

    assert.ok(gold, "gold must be in results");
    assert.equal(gold!.vecRank, 1, "gold should be vec rank 1");
    assert.equal(gold!.ftsRank, null, "gold should have no FTS rank (not in fts leg)");

    assert.ok(decoy, "decoy must be in results");
    assert.equal(decoy!.ftsRank, 1, "decoy should be fts rank 1");
    assert.equal(decoy!.vecRank, null, "decoy should have no vec rank (outside legK=1)");
  });

  it("RRF scores are strictly positive", async () => {
    const results = await hybridRetrieve("u_read", "morning routine habits", db, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
      legK: 1,
    });
    for (const r of results) {
      assert.ok(r.rrfScore > 0, `rrfScore must be positive, got ${r.rrfScore}`);
    }
  });

  it("result order is deterministic: same inputs → same ranked output", async () => {
    const opts = { queryEmbedding: unitVec(0), skipRerank: true as const, legK: 5, k: 5 };
    const run1 = await hybridRetrieve("u_read", "morning routine habits", db, opts);
    const run2 = await hybridRetrieve("u_read", "morning routine habits", db, opts);
    assert.deepEqual(
      run1.map((r) => r.id),
      run2.map((r) => r.id),
      "two identical calls must return the same order",
    );
  });

  it("access_count increments for returned memories", async () => {
    const before = getMemory(db, goldId)!.access_count;
    await hybridRetrieve("u_read", "morning routine habits", db, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
      legK: 1,
      k: 5,
    });
    const after = getMemory(db, goldId)!.access_count;
    assert.ok(after > before, "access_count should increment for retrieved memories");
  });

  it("user isolation: results contain only memories for the querying user", async () => {
    insertMemory(db, {
      user_id: "u_other",
      type: "fact",
      content: "This belongs to another user entirely.",
      embedding: unitVec(0),
    });

    const results = await hybridRetrieve("u_read", "morning routine habits", db, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
      legK: 5,
    });

    for (const r of results) {
      const m = getMemory(db, r.id)!;
      assert.equal(m.user_id, "u_read", `memory ${r.id} belongs to wrong user: ${m.user_id}`);
    }
  });

  it("empty store returns empty list gracefully", async () => {
    const db3 = openDb(":memory:");
    const results = await hybridRetrieve("u_empty", "anything", db3, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
    });
    assert.equal(results.length, 0);
    db3.close();
  });

  it("rerankScore is null when rerank is skipped", async () => {
    const results = await hybridRetrieve("u_read", "morning routine habits", db, {
      queryEmbedding: unitVec(0),
      skipRerank: true,
      legK: 2,
    });
    for (const r of results) {
      assert.equal(r.rerankScore, null, "rerankScore should be null when skipped");
    }
  });
});

// ─── PART A2: SPEC §4.1 paraphrase case — "dim interfaces" vs "dark mode" ─────
//
// gold:   "The user prefers dim, low-brightness interfaces." — zero lexical
//         overlap with the query "dark mode", but semantically the answer.
// decoy:  "…watched a video called Dark Mode…" — exact lexical match, wrong meaning.
// filler: pushes the decoy out of the vec leg so FTS is its only entry point.

describe("paraphrase retrieval: 'dark mode' query (offline)", () => {
  let db: Db;
  let goldId: string;
  let decoyId: string;

  const queryVec = unitVec(0);

  before(async () => {
    db = openDb(":memory:");
    goldId = insertMemory(db, {
      user_id: "u_dm",
      type: "preference",
      content: "The user prefers dim, low-brightness interfaces in every app.",
      salience: 0.8,
      embedding: nearVec(0, 0.95),   // semantically ≈ the query
    });
    decoyId = insertMemory(db, {
      user_id: "u_dm",
      type: "note",
      content: "The user watched a video called Dark Mode about cave diving.",
      salience: 0.4,
      embedding: unitVec(700),       // semantically unrelated
    });
    // Fillers semantically closer to the query than the decoy — with legK=3
    // they crowd the decoy out of the vec leg entirely.
    insertMemory(db, {
      user_id: "u_dm", type: "note", salience: 0.3,
      content: "The user reads documentation at night.",
      embedding: nearVec(0, 0.75),
    });
    insertMemory(db, {
      user_id: "u_dm", type: "note", salience: 0.3,
      content: "The user keeps screen brightness low to reduce eye strain.",
      embedding: nearVec(0, 0.8),
    });
  });

  after(() => { db.close(); });

  it("FTS alone ranks the decoy first and misses the paraphrase", () => {
    const fts = ftsSearch(db, "u_dm", "dark mode", 10);
    assert.equal(fts[0]?.id, decoyId, "lexical leg must find the decoy");
    assert.ok(!fts.some((r) => r.id === goldId), "lexical leg cannot see the paraphrase");
  });

  it("hybrid ranks the semantic match above the lexical decoy", async () => {
    const results = await hybridRetrieve("u_dm", "dark mode", db, {
      queryEmbedding: queryVec,
      legK: 3,
      skipRerank: true,
    });
    const goldPos  = results.findIndex((r) => r.id === goldId);
    const decoyPos = results.findIndex((r) => r.id === decoyId);
    assert.ok(goldPos >= 0, "gold must be retrieved");
    assert.ok(decoyPos >= 0, "decoy still surfaces (hybrid, not vec-only)");
    assert.ok(goldPos < decoyPos, `gold (#${goldPos + 1}) must outrank decoy (#${decoyPos + 1})`);
    // Provenance: gold arrived via the vec leg only, decoy via the FTS leg only.
    assert.equal(results[goldPos].ftsRank, null);
    assert.equal(results[goldPos].vecRank, 1);
    assert.equal(results[decoyPos].ftsRank, 1);
  });

  it("RRF determinism: identical inputs produce identical ranked output", async () => {
    const run = () =>
      hybridRetrieve("u_dm", "dark mode", db, {
        queryEmbedding: queryVec, legK: 3, skipRerank: true,
      });
    const a = await run();
    const b = await run();
    assert.deepEqual(
      b.map((r) => ({ id: r.id, score: r.rrfScore })),
      a.map((r) => ({ id: r.id, score: r.rrfScore })),
    );
  });
});

// ─── PART A3: rerank (offline — qwen-flash mocked via fetch stub) ─────────────

describe("rerank (offline, mocked)", () => {
  let db: Db;
  const realFetch = globalThis.fetch;

  before(() => { db = openDb(":memory:"); });
  after(() => {
    db.close();
    globalThis.fetch = realFetch;
  });

  function mockRerankScores(fn: (ids: string[]) => Record<string, number>) {
    globalThis.fetch = (async (_url: any, init?: any) => {
      const body = JSON.parse(String(init?.body));
      const userMsg: string = body.messages[1].content;
      const ids = [...userMsg.matchAll(/id=(\S+)/g)].map((m) => m[1]);
      const scores = fn(ids);
      return new Response(
        JSON.stringify({
          id: "chatcmpl-test",
          object: "chat.completion",
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: JSON.stringify({
                scores: ids.map((id) => ({ id, score: scores[id] ?? 0 })),
              }),
            },
            finish_reason: "stop",
          }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;
  }

  it("re-sorts by qwen-flash score and populates rerankScore", async () => {
    const gold = insertMemory(db, {
      user_id: "u_rr", type: "preference", salience: 0.8,
      content: "The user prefers dim, low-brightness interfaces.",
      embedding: nearVec(0, 0.95),
    });
    const decoy = insertMemory(db, {
      user_id: "u_rr", type: "note", salience: 0.4,
      content: "The user watched a video called Dark Mode about cave diving.",
      embedding: nearVec(0, 0.9),   // vec-close too: RRF alone would rank it #2
    });

    mockRerankScores((_ids) => ({ [gold]: 9, [decoy]: 1 }));

    const results = await hybridRetrieve("u_rr", "dark mode", db, {
      queryEmbedding: unitVec(0),
      legK: 5,
      // rerank NOT skipped — goes through the mocked flash call
    });

    assert.equal(results[0].id, gold);
    assert.equal(results[0].rerankScore, 9);
    const decoyRow = results.find((r) => r.id === decoy);
    assert.equal(decoyRow?.rerankScore, 1);
  });

  it("rerank failure degrades gracefully to RRF order", async () => {
    globalThis.fetch = (async () =>
      new Response("boom", { status: 500 })) as typeof fetch;

    const results = await hybridRetrieve("u_rr", "dark mode", db, {
      queryEmbedding: unitVec(0),
      legK: 5,
    });
    assert.ok(results.length > 0, "results still returned");
    for (const r of results) assert.equal(r.rerankScore, null);
  });
});

// ─── PART B: integration tests (requires real DASHSCOPE_API_KEY) ─────────────

const _rawKey = process.env.DASHSCOPE_API_KEY ?? "";
const HAS_API_KEY = _rawKey.length > 20 && !_rawKey.includes("...");

describe("hybridRetrieve + rerank (integration)", { skip: !HAS_API_KEY }, () => {
  let db: Db;

  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  let ids: Record<string, string>;

  before(() => {
    ids = {
      pref:  insertMemory(db, { user_id: "u_int", type: "preference",
        content: "The user prefers concise, bullet-point answers over long prose.",
        salience: 0.85, embedding: unitVec(0) }),
      fact:  insertMemory(db, { user_id: "u_int", type: "fact",
        content: "The user has been programming in TypeScript for five years.",
        salience: 0.6,  embedding: unitVec(1) }),
      skill: insertMemory(db, { user_id: "u_int", type: "skill",
        content: "The user is proficient with Docker and container orchestration.",
        salience: 0.7,  embedding: unitVec(2) }),
      noise: insertMemory(db, { user_id: "u_int", type: "note",
        content: "The user's cat is named Whiskers.",
        salience: 0.2,  embedding: unitVec(3) }),
    };
  });

  it("live hybridRetrieve: real embed + FTS + vec, results include relevant memories", async () => {
    const results = await hybridRetrieve("u_int", "TypeScript programming experience", db, {
      skipRerank: true,
      k: 5,
    });

    console.log("\n--- hybridRetrieve results (live embed, no rerank) ---");
    for (const r of results) {
      console.log(
        `  [fts=${r.ftsRank ?? "-"} vec=${r.vecRank ?? "-"} rrf=${r.rrfScore.toFixed(5)}] ${r.content.slice(0, 70)}`
      );
    }
    console.log("--- end results ---\n");

    assert.ok(results.length > 0, "should return at least one result");
    assert.ok(results.every((r) => r.rrfScore > 0), "all RRF scores must be positive");
  });

  it("rerank: rerankScore is populated and list is sorted descending", async () => {
    const candidates = await hybridRetrieve("u_int", "response format preferences", db, {
      skipRerank: true,
      k: 4,
    });

    if (candidates.length < 2) return;

    const reranked = await rerank("response format preferences", candidates);

    console.log("\n--- reranked results ---");
    for (const r of reranked) {
      console.log(`  [rerank=${r.rerankScore ?? "-"} rrf=${r.rrfScore.toFixed(5)}] ${r.content.slice(0, 70)}`);
    }
    console.log("--- end reranked results ---\n");

    assert.ok(reranked.some((r) => r.rerankScore !== null), "at least one rerankScore should be set");
    for (let i = 1; i < reranked.length; i++) {
      const a = reranked[i - 1].rerankScore ?? 0;
      const b = reranked[i].rerankScore ?? 0;
      assert.ok(a >= b, `reranked list not sorted at positions ${i - 1}/${i}: ${a} < ${b}`);
    }
  });
});
