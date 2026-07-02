/**
 * test/write.test.ts
 *
 * Write path tests:
 *   PART A — unit tests (no API key needed): writeMemories with pre-baked embeddings.
 *   PART B — integration tests (requires DASHSCOPE_API_KEY): live extractMemories call.
 *
 * Run: npm test
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openDb, type Db } from "../src/db/client.js";
import { listMemories, getMemory, edgesFor } from "../src/db/queries.js";
import { extractMemories, writeMemories, type MemoryCandidate } from "../src/memory/write.js";

// ─── Synthetic embedding helpers ─────────────────────────────────────────────

/** Unit vector along dimension d (all other dims 0). */
function unitVec(d: number, dim = 1024): number[] {
  const v = new Array<number>(dim).fill(0);
  v[d] = 1.0;
  return v;
}

/**
 * Return a vector with cosine similarity `sim` to unitVec(d).
 * Achieved by mixing unitVec(d) and unitVec(d+1), then L2-normalising.
 */
function nearVec(d: number, sim: number, dim = 1024): number[] {
  // cos θ = sim  →  component along d+1 = √(1 - sim²)
  const perp = Math.sqrt(1 - sim * sim);
  const v = new Array<number>(dim).fill(0);
  v[d] = sim;
  v[d + 1] = perp;
  return v;
}

// ─── PART A: unit tests (offline) ────────────────────────────────────────────

describe("writeMemories (offline)", () => {
  let db: Db;

  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  it("inserts two distinct candidates and returns their ids", async () => {
    const candidates: MemoryCandidate[] = [
      {
        type: "preference",
        content: "The user prefers TypeScript over JavaScript for all projects.",
        salience: 0.85,
        confidence: 1.0,
        tags: ["typescript", "preferences"],
        embedding: unitVec(100),
      },
      {
        type: "fact",
        content: "The user works as a senior software engineer at a startup.",
        salience: 0.6,
        confidence: 0.9,
        tags: ["career", "facts"],
        embedding: unitVec(200),
      },
    ];

    const result = await writeMemories("u1", "s-test", candidates, db);

    assert.equal(result.inserted.length, 2, "expected 2 insertions");
    assert.equal(result.skipped.length, 0, "expected 0 skips");

    const stored = listMemories(db, { user_id: "u1", status: "active" });
    assert.equal(stored.length, 2, "DB should have exactly 2 active memories");
    assert.ok(stored.some((m) => m.type === "preference"), "preference memory missing");
    assert.ok(stored.some((m) => m.type === "fact"), "fact memory missing");
  });

  it("exact duplicate: skips on re-run and bumps access_count", async () => {
    // Write same two candidates again
    const candidates: MemoryCandidate[] = [
      {
        type: "preference",
        content: "The user prefers TypeScript over JavaScript for all projects.",
        salience: 0.85,
        confidence: 1.0,
        tags: ["typescript", "preferences"],
        embedding: unitVec(100),
      },
      {
        type: "fact",
        content: "The user works as a senior software engineer at a startup.",
        salience: 0.6,
        confidence: 0.9,
        tags: ["career", "facts"],
        embedding: unitVec(200),
      },
    ];

    const before = listMemories(db, { user_id: "u1", status: "active" });
    const result = await writeMemories("u1", "s-test", candidates, db);
    const after = listMemories(db, { user_id: "u1", status: "active" });

    assert.equal(result.inserted.length, 0, "no new insertions expected");
    assert.equal(result.skipped.length, 2, "both should be exact-match skips");
    assert.ok(result.skipped.every((s) => s.reason === "exact_match"), "wrong skip reason");
    assert.equal(before.length, after.length, "total memory count must not change");
  });

  it("semantic duplicate: skips a near-identical candidate (cosine ≥ 0.95)", async () => {
    // Insert an anchor at unitVec(300)
    await writeMemories("u_sem", "s-test", [
      {
        type: "fact",
        content: "The user drinks coffee every morning.",
        salience: 0.5,
        confidence: 0.9,
        tags: ["habits"],
        embedding: unitVec(300),
      },
    ], db);

    // Try to insert something with cosine similarity = 0.97 to the anchor
    const result = await writeMemories("u_sem", "s-test", [
      {
        type: "fact",
        content: "The user has coffee every morning.",  // slightly different wording
        salience: 0.5,
        confidence: 0.9,
        tags: ["habits"],
        embedding: nearVec(300, 0.97),   // cosine dist = 0.03 < threshold 0.05
      },
    ], db);

    assert.equal(result.skipped.length, 1, "near-duplicate should be skipped");
    assert.equal(result.skipped[0].reason, "semantic_duplicate");
    assert.equal(result.inserted.length, 0, "no new memories should be inserted");
  });

  it("semantic near-miss: inserts a candidate below the similarity threshold", async () => {
    // Insert an anchor at unitVec(400)
    await writeMemories("u_miss", "s-test", [
      {
        type: "fact",
        content: "The user likes hiking in the mountains.",
        salience: 0.5,
        confidence: 0.9,
        tags: ["hobbies"],
        embedding: unitVec(400),
      },
    ], db);

    // cosine similarity = 0.70 → distance = 0.30 > threshold 0.05 → NOT a dupe
    const result = await writeMemories("u_miss", "s-test", [
      {
        type: "fact",
        content: "The user enjoys cycling on weekends.",
        salience: 0.5,
        confidence: 0.9,
        tags: ["hobbies"],
        embedding: nearVec(400, 0.70),   // cosine dist = 0.30 > 0.05
      },
    ], db);

    assert.equal(result.inserted.length, 1, "dissimilar candidate should be inserted");
    assert.equal(result.skipped.length, 0);
  });

  it("user isolation: dedup does not cross user boundaries", async () => {
    const sharedContent = "The user prefers remote work.";
    const emb = unitVec(500);

    // Write for user A
    await writeMemories("u_iso_a", "s-test", [
      { type: "preference", content: sharedContent, salience: 0.8, confidence: 1.0, tags: [], embedding: emb },
    ], db);

    // Same content + near-identical embedding for user B should still be inserted
    const result = await writeMemories("u_iso_b", "s-test", [
      { type: "preference", content: sharedContent, salience: 0.8, confidence: 1.0, tags: [], embedding: emb },
    ], db);

    assert.equal(result.inserted.length, 1, "same content for different user should insert");
  });

  it("returns correct tags as parsed JSON", async () => {
    const result = await writeMemories("u_tags", "s-test", [
      {
        type: "skill",
        content: "The user is proficient in Python and data science.",
        salience: 0.7,
        confidence: 0.95,
        tags: ["python", "data-science", "skills"],
        embedding: unitVec(600),
      },
    ], db);

    assert.equal(result.inserted.length, 1);
    const stored = listMemories(db, { user_id: "u_tags" });
    const tags = JSON.parse(stored[0].tags) as string[];
    assert.deepEqual(tags, ["python", "data-science", "skills"]);
  });
});

// ─── PART A2: write-time supersession (offline — LLM verdict mocked) ─────────
// The qwen client resolves globalThis.fetch per call, so stubbing it here
// intercepts the supersession chatJSON without any network access.

describe("write-time supersession (offline, mocked verdict)", () => {
  let db: Db;
  const realFetch = globalThis.fetch;

  before(() => { db = openDb(":memory:"); });
  after(() => {
    db.close();
    globalThis.fetch = realFetch;
  });

  function mockVerdict(verdicts: object[]) {
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({
          id: "chatcmpl-test",
          object: "chat.completion",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: JSON.stringify({ verdicts }) },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      )) as typeof fetch;
  }

  it('"I use Postgres" → "I switched to SQLite": old superseded + edge + belief event', async () => {
    // Session 1: the original decision.
    const first = await writeMemories("u_ss", "session-1", [
      {
        type: "decision",
        content: "The user uses Postgres for the project database.",
        salience: 0.8,
        confidence: 0.95,
        tags: ["database"],
        embedding: unitVec(300),
      },
    ], db);
    assert.equal(first.inserted.length, 1);
    const oldId = first.inserted[0];

    // Session 2: the change of mind. Similar embedding (cos 0.90) so the old
    // memory is the nearest neighbour, but below the 0.95 dedupe threshold.
    mockVerdict([{ old_index: 1, action: "update", reasoning: "The user switched databases." }]);
    const second = await writeMemories("u_ss", "session-2", [
      {
        type: "decision",
        content: "The user switched the project database to SQLite.",
        salience: 0.85,
        confidence: 0.95,
        tags: ["database"],
        embedding: nearVec(300, 0.9),
      },
    ], db);
    assert.equal(second.inserted.length, 1, "new memory must insert, not dedupe");
    const newId = second.inserted[0];

    // Old memory: superseded, points at its successor.
    const oldMem = getMemory(db, oldId);
    assert.equal(oldMem?.status, "superseded");
    assert.equal(oldMem?.superseded_by, newId);

    // Graph: supersedes edge new → old.
    const edges = edgesFor(db, newId);
    assert.equal(edges.length, 1);
    assert.deepEqual(
      { src: edges[0].src_id, dst: edges[0].dst_id, kind: edges[0].kind },
      { src: newId, dst: oldId, kind: "supersedes" },
    );

    // Audit trail: belief event on the old memory, session provenance intact.
    const events = db
      .prepare("SELECT * FROM belief_events WHERE memory_id = ?")
      .all(oldId) as { delta: number; reason: string; turn_id: string }[];
    assert.equal(events.length, 1);
    assert.equal(events[0].reason, "contradicted");
    assert.ok(events[0].delta < 0, "contradiction must carry a negative delta");
    assert.equal(events[0].turn_id, "session-2");

    // Provenance: each memory remembers the session it was born in.
    assert.equal(oldMem?.session_id, "session-1");
    assert.equal(getMemory(db, newId)?.session_id, "session-2");
  });

  it("unrelated verdict leaves both memories active", async () => {
    mockVerdict([]); // first write: no neighbours → no LLM call anyway
    const a = await writeMemories("u_nr", "s1", [
      {
        type: "fact",
        content: "The user lives in New Jersey.",
        salience: 0.5, confidence: 0.9, tags: [],
        embedding: unitVec(500),
      },
    ], db);

    mockVerdict([{ old_index: 1, action: "unrelated", reasoning: "Different topics." }]);
    const b = await writeMemories("u_nr", "s2", [
      {
        type: "fact",
        content: "The user's favorite editor is Neovim.",
        salience: 0.5, confidence: 0.9, tags: [],
        embedding: nearVec(500, 0.8),
      },
    ], db);

    assert.equal(getMemory(db, a.inserted[0])?.status, "active");
    assert.equal(getMemory(db, b.inserted[0])?.status, "active");
    assert.equal(edgesFor(db, b.inserted[0]).length, 0, "no edge for unrelated");
  });
});

// ─── PART B: integration tests (requires DASHSCOPE_API_KEY) ──────────────────

// Require a real key (not the placeholder copied from .env.example)
const _rawKey = process.env.DASHSCOPE_API_KEY ?? "";
const HAS_API_KEY = _rawKey.length > 20 && !_rawKey.includes("...");

describe("extractMemories + writeMemories (integration)", { skip: !HAS_API_KEY }, () => {
  let db: Db;

  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  const SAMPLE_TURN = `\
I'm building a new TypeScript project and I'd really appreciate your help.
I always prefer to use strict mode and functional patterns when possible.
I work mainly in VS Code on macOS. I've been doing this for about 8 years now.
My current project is a real-time chat app — can you suggest a good WebSocket library?`;

  it("extractMemories: returns typed memories from a sample turn", async () => {
    const memories = await extractMemories(SAMPLE_TURN);

    console.log("\n--- Extracted JSON ---");
    console.log(JSON.stringify(memories, null, 2));
    console.log("--- End extracted JSON ---\n");

    assert.ok(Array.isArray(memories), "result should be an array");
    assert.ok(memories.length >= 2, `expected ≥ 2 memories, got ${memories.length}`);

    // Must have at least one preference
    assert.ok(
      memories.some((m) => m.type === "preference"),
      `no preference found; types: ${memories.map((m) => m.type).join(", ")}`
    );

    // Must have at least one fact
    assert.ok(
      memories.some((m) => m.type === "fact"),
      `no fact found; types: ${memories.map((m) => m.type).join(", ")}`
    );

    for (const m of memories) {
      assert.ok(m.content.length >= 10, "content too short");
      assert.ok(m.salience >= 0 && m.salience <= 1, "salience out of range");
      assert.ok(m.confidence >= 0 && m.confidence <= 1, "confidence out of range");
      assert.ok(Array.isArray(m.tags), "tags should be array");
      // Self-contained: should NOT start with a pronoun
      const firstWord = m.content.split(" ")[0].toLowerCase();
      assert.ok(
        !["i", "my", "me", "we", "they", "he", "she", "it"].includes(firstWord),
        `content starts with pronoun '${firstWord}': "${m.content.slice(0, 80)}"`
      );
    }
  });

  it("end-to-end: extract + writeMemories stores preference and fact with embeddings", async () => {
    const extracted = await extractMemories(SAMPLE_TURN);
    const result = await writeMemories("u_e2e", "s-test", extracted, db);

    console.log(`\nInserted ${result.inserted.length}, skipped ${result.skipped.length}`);

    assert.ok(result.inserted.length >= 2, "expected ≥ 2 inserts");

    const stored = listMemories(db, { user_id: "u_e2e", status: "active" });
    assert.ok(stored.some((m) => m.type === "preference"), "preference not stored");
    assert.ok(stored.some((m) => m.type === "fact"), "fact not stored");
    // All should have embeddings (1024-dim)
    for (const m of stored) {
      assert.ok(m.embedding !== null, `memory ${m.id} has no embedding`);
      assert.equal(m.embedding!.length, 1024, "wrong embedding dim");
    }
  });

  it("idempotent: re-running same extraction does not create duplicates", async () => {
    const extracted = await extractMemories(SAMPLE_TURN);

    const first = await writeMemories("u_idem", "s-test", extracted, db);
    const countAfterFirst = listMemories(db, { user_id: "u_idem", status: "active" }).length;

    const second = await writeMemories("u_idem", "s-test", extracted, db);
    const countAfterSecond = listMemories(db, { user_id: "u_idem", status: "active" }).length;

    assert.equal(countAfterFirst, countAfterSecond, "second run must not add new memories");
    assert.equal(second.inserted.length, 0, "second run should insert nothing");
    assert.ok(
      second.skipped.length > 0,
      `second run should have skips; got ${second.skipped.length}`
    );
    console.log(
      `\nIdempotency: first=${first.inserted.length} inserts, ` +
      `second=${second.skipped.length} skips (${second.skipped.map((s) => s.reason).join(", ")})`
    );
  });
});
