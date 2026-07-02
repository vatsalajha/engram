/**
 * test/db.test.ts
 *
 * Database layer tests — uses an in-memory SQLite database (no env vars needed).
 * Run: npm test
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openDb, closeDb, type Db } from "../src/db/client.js";
import {
  insertMemory,
  getMemory,
  updateMemoryStatus,
  touchMemory,
  setConfidence,
  contentHash,
  addEdge,
  edgesFor,
  insertBeliefEvent,
  insertSleepRun,
  ftsSearch,
  vecSearch,
  listMemories,
  insertEpisodic,
  recentEpisodic,
  insertEvalRun,
  insertConflict,
  insertConsolidation,
} from "../src/db/queries.js";

// ─── Fixed 1024-dim test embeddings ──────────────────────────────────────────

/** Make a unit vector along dimension `d` (all others 0). */
function unitVec(d: number, dim = 1024): number[] {
  const v = new Array<number>(dim).fill(0);
  v[d] = 1.0;
  return v;
}

// ─── Test suite ──────────────────────────────────────────────────────────────

describe("Engram DB layer", () => {
  let db: Db;

  before(() => {
    db = openDb(":memory:");
  });

  after(() => {
    db.close();
  });

  // ── Schema smoke ────────────────────────────────────────────────────────────

  it("schema: all tables exist", () => {
    const tables = (
      db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all() as { name: string }[]
    ).map((r) => r.name);

    for (const t of ["memories", "episodic_log", "conflicts", "consolidations", "eval_runs"]) {
      assert.ok(tables.includes(t), `table '${t}' missing`);
    }
  });

  it("schema: virtual tables exist", () => {
    const vtables = (
      db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all() as { name: string }[]
    ).map((r) => r.name);
    assert.ok(vtables.includes("memories_fts"), "memories_fts missing");

    // vec0 tables appear as 'shadow' tables with a specific prefix
    const names = (
      db.prepare("SELECT name FROM sqlite_master").all() as { name: string }[]
    ).map((r) => r.name);
    assert.ok(names.some((n) => n.startsWith("vec_memories")), "vec_memories missing");
  });

  it("schema: FTS triggers exist", () => {
    const triggers = (
      db
        .prepare("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='memories'")
        .all() as { name: string }[]
    ).map((r) => r.name);

    assert.ok(triggers.includes("memories_fts_insert"), "insert trigger missing");
    assert.ok(triggers.includes("memories_fts_delete"), "delete trigger missing");
    assert.ok(triggers.includes("memories_fts_update"), "update trigger missing");
  });

  // ── insertMemory / getMemory ─────────────────────────────────────────────────

  it("insertMemory: returns a ULID and memory is retrievable", () => {
    const id = insertMemory(db, {
      user_id: "u1",
      type: "preference",
      content: "User prefers dark mode in all applications.",
      salience: 0.9,
    });

    assert.match(id, /^[0-9A-Z]{26}$/, "id should be a ULID");

    const mem = getMemory(db, id);
    assert.ok(mem, "memory not found");
    assert.equal(mem.content, "User prefers dark mode in all applications.");
    assert.equal(mem.type, "preference");
    assert.equal(mem.status, "active");
    assert.equal(mem.salience, 0.9);
  });

  it("insertMemory: stores embedding in both memories and vec_memories", () => {
    const embedding = unitVec(0); // [1, 0, 0, ...]
    const id = insertMemory(db, {
      user_id: "u1",
      type: "fact",
      content: "TypeScript is a typed superset of JavaScript.",
      embedding,
    });

    // Check memories.embedding is stored
    const mem = getMemory(db, id);
    assert.ok(mem?.embedding, "embedding not stored in memories");
    assert.equal(mem!.embedding!.length, 1024);
    assert.ok(Math.abs(mem!.embedding![0] - 1.0) < 1e-5, "first component should be ~1");

    // Check vec_memories has the row
    const vecRow = db
      .prepare("SELECT memory_id FROM vec_memories WHERE memory_id = ?")
      .get(id) as { memory_id: string } | undefined;
    assert.ok(vecRow, "vec_memories row missing");
  });

  // ── FTS search ──────────────────────────────────────────────────────────────

  it("ftsSearch: finds memory by keyword", () => {
    const id = insertMemory(db, {
      user_id: "u_fts",
      type: "preference",
      content: "User always greets colleagues with a formal handshake.",
    });

    const results = ftsSearch(db, "u_fts", "handshake");
    assert.ok(results.length > 0, "FTS returned no results");
    assert.equal(results[0].id, id);
  });

  it("ftsSearch: Porter stemming finds root forms", () => {
    insertMemory(db, {
      user_id: "u_stem",
      type: "fact",
      content: "The user is currently running three projects simultaneously.",
    });

    // 'running' and 'run' should both hit via Porter stemming
    const r1 = ftsSearch(db, "u_stem", "run");
    const r2 = ftsSearch(db, "u_stem", "running");
    assert.ok(r1.length > 0, "stemmed query 'run' found nothing");
    assert.ok(r2.length > 0, "exact query 'running' found nothing");
  });

  it("ftsSearch: scoped by user_id", () => {
    insertMemory(db, { user_id: "u_scope_a", type: "fact", content: "The sky is blue and clear." });
    insertMemory(db, { user_id: "u_scope_b", type: "fact", content: "The sky is green and foggy." });

    const a = ftsSearch(db, "u_scope_a", "sky");
    const b = ftsSearch(db, "u_scope_b", "sky");
    assert.ok(a.every((r) => r.content.includes("blue")), "u_scope_b leak into u_scope_a");
    assert.ok(b.every((r) => r.content.includes("green")), "u_scope_a leak into u_scope_b");
  });

  // ── FTS trigger correctness ──────────────────────────────────────────────────

  it("FTS trigger: update removes old terms and indexes new content", () => {
    const id = insertMemory(db, {
      user_id: "u_trig",
      type: "preference",
      content: "User prefers coffee in the morning.",
    });

    // Confirm 'coffee' is found
    let r = ftsSearch(db, "u_trig", "coffee");
    assert.ok(r.some((x) => x.id === id), "coffee not indexed before update");

    // Update content
    db.prepare("UPDATE memories SET content = ? WHERE id = ?").run(
      "User prefers tea in the morning.",
      id
    );

    // 'coffee' should no longer match
    r = ftsSearch(db, "u_trig", "coffee");
    assert.ok(!r.some((x) => x.id === id), "coffee still indexed after update");

    // 'tea' should match
    r = ftsSearch(db, "u_trig", "tea");
    assert.ok(r.some((x) => x.id === id), "tea not indexed after update");
  });

  it("FTS trigger: delete removes document from index", () => {
    const id = insertMemory(db, {
      user_id: "u_del",
      type: "fact",
      content: "User owns a golden retriever named Biscuit.",
    });

    let r = ftsSearch(db, "u_del", "Biscuit");
    assert.ok(r.some((x) => x.id === id), "not found before delete");

    db.prepare("DELETE FROM memories WHERE id = ?").run(id);

    r = ftsSearch(db, "u_del", "Biscuit");
    assert.ok(!r.some((x) => x.id === id), "still found after delete");
  });

  // ── Vector search ────────────────────────────────────────────────────────────

  it("vecSearch: returns nearest neighbours by cosine distance", () => {
    const userId = "u_vec";

    // Three orthogonal unit vectors
    const id0 = insertMemory(db, {
      user_id: userId,
      type: "fact",
      content: "Vector alpha.",
      embedding: unitVec(0),
    });
    const id1 = insertMemory(db, {
      user_id: userId,
      type: "fact",
      content: "Vector beta.",
      embedding: unitVec(1),
    });
    const id2 = insertMemory(db, {
      user_id: userId,
      type: "fact",
      content: "Vector gamma.",
      embedding: unitVec(2),
    });

    // Query close to unitVec(0) — id0 should be nearest
    const results = vecSearch(db, userId, unitVec(0), 3);
    assert.ok(results.length >= 1, "vecSearch returned nothing");
    assert.equal(results[0].id, id0, "nearest neighbour should be id0");
    assert.ok(results[0].distance < 0.01, "exact match should have near-zero distance");

    // id1 and id2 should follow (distance == √2 for orthogonal unit vecs)
    const ids = results.map((r) => r.id);
    assert.ok(ids.includes(id1), "id1 missing from results");
    assert.ok(ids.includes(id2), "id2 missing from results");
  });

  it("vecSearch: scoped by user_id", () => {
    const emb = unitVec(10);
    insertMemory(db, { user_id: "u_vec_a", type: "fact", content: "A", embedding: emb });
    insertMemory(db, { user_id: "u_vec_b", type: "fact", content: "B", embedding: emb });

    const a = vecSearch(db, "u_vec_a", emb, 10);
    assert.ok(a.every((r) => r.id !== undefined), "bad result shape");
    // Confirm u_vec_b's record does not appear in u_vec_a's results
    const ids = a.map((r) => r.id);
    const leaked = db
      .prepare("SELECT id FROM memories WHERE user_id='u_vec_b'")
      .all() as { id: string }[];
    for (const l of leaked) {
      assert.ok(!ids.includes(l.id), `u_vec_b memory ${l.id} leaked into u_vec_a results`);
    }
  });

  it("vecSearch: excludes archived/superseded memories", () => {
    const userId = "u_vec_status";
    const emb = unitVec(20);

    const activeId = insertMemory(db, {
      user_id: userId, type: "fact", content: "Active.", embedding: emb,
    });
    const archivedId = insertMemory(db, {
      user_id: userId, type: "fact", content: "Archived.", embedding: emb,
    });

    updateMemoryStatus(db, archivedId, "archived");

    const results = vecSearch(db, userId, emb, 10);
    const ids = results.map((r) => r.id);
    assert.ok(ids.includes(activeId), "active memory missing from results");
    assert.ok(!ids.includes(archivedId), "archived memory should be excluded");
  });

  // ── updateMemoryStatus / touchMemory ─────────────────────────────────────────

  it("updateMemoryStatus: changes status and superseded_by", () => {
    const old = insertMemory(db, { user_id: "u_upd", type: "fact", content: "Old fact." });
    const newer = insertMemory(db, { user_id: "u_upd", type: "fact", content: "New fact." });

    updateMemoryStatus(db, old, "superseded", newer);

    const mem = getMemory(db, old);
    assert.equal(mem?.status, "superseded");
    assert.equal(mem?.superseded_by, newer);
  });

  it("touchMemory: increments access_count and updates last_accessed_at", async () => {
    const id = insertMemory(db, { user_id: "u_touch", type: "fact", content: "Touched." });
    const before = getMemory(db, id)!;
    assert.equal(before.access_count, 0);

    // Small delay so last_accessed_at changes
    await new Promise((r) => setTimeout(r, 5));

    touchMemory(db, id);
    const after = getMemory(db, id)!;
    assert.equal(after.access_count, 1);
    assert.ok(after.last_accessed_at >= before.last_accessed_at);

    touchMemory(db, id);
    const after2 = getMemory(db, id)!;
    assert.equal(after2.access_count, 2);
  });

  // ── listMemories ─────────────────────────────────────────────────────────────

  it("listMemories: filters by type", () => {
    insertMemory(db, { user_id: "u_list", type: "preference", content: "A pref." });
    insertMemory(db, { user_id: "u_list", type: "fact", content: "A fact." });

    const prefs = listMemories(db, { user_id: "u_list", type: "preference" });
    assert.ok(prefs.every((m) => m.type === "preference"));
    assert.ok(prefs.length >= 1);
  });

  // ── Supporting tables ─────────────────────────────────────────────────────────

  it("insertEpisodic: stores a turn", () => {
    const id = insertEpisodic(db, {
      user_id: "u_ep",
      session_id: "sess_1",
      role: "user",
      content: "Hello, remember I like jazz.",
    });
    assert.match(id, /^[0-9A-Z]{26}$/);

    const row = db
      .prepare("SELECT * FROM episodic_log WHERE id=?")
      .get(id) as { role: string; content: string } | undefined;
    assert.ok(row);
    assert.equal(row.role, "user");
  });

  it("insertConflict: records a contradiction", () => {
    const m1 = insertMemory(db, { user_id: "u_cf", type: "fact", content: "Fact A." });
    const m2 = insertMemory(db, { user_id: "u_cf", type: "fact", content: "Fact B contradicts A." });

    const cid = insertConflict(db, {
      new_id: m2,
      old_id: m1,
      resolution: "superseded",
      reasoning: "B explicitly updates A.",
    });

    const row = db
      .prepare("SELECT * FROM conflicts WHERE id=?")
      .get(cid) as { resolution: string; reasoning: string } | undefined;
    assert.ok(row);
    assert.equal(row.resolution, "superseded");
  });

  it("insertConsolidation: records a merge", () => {
    const summary = insertMemory(db, {
      user_id: "u_con",
      type: "fact",
      content: "Summary of merged facts.",
      source: "consolidated",
    });
    const src1 = insertMemory(db, { user_id: "u_con", type: "event", content: "Event 1." });
    const src2 = insertMemory(db, { user_id: "u_con", type: "event", content: "Event 2." });

    const cid = insertConsolidation(db, { summary_id: summary, source_ids: [src1, src2] });

    const row = db
      .prepare("SELECT * FROM consolidations WHERE id=?")
      .get(cid) as { source_ids: string } | undefined;
    assert.ok(row);
    const parsed = JSON.parse(row.source_ids) as string[];
    assert.deepEqual(parsed, [src1, src2]);
  });

  // ── v2: content-hash idempotency ────────────────────────────────────────────

  it("content_hash: stored on insert, stable sha256 of content", () => {
    const id = insertMemory(db, { user_id: "u_hash", type: "fact", content: "Hash me." });
    const mem = getMemory(db, id);
    assert.ok(mem);
    assert.equal(mem.content_hash, contentHash("Hash me."));
    assert.match(mem.content_hash, /^[0-9a-f]{64}$/);
  });

  it("content_hash: duplicate insert is graceful — same id back, no second row", () => {
    const first = insertMemory(db, {
      user_id: "u_dup",
      type: "preference",
      content: "User prefers tabs over spaces.",
    });
    const second = insertMemory(db, {
      user_id: "u_dup",
      type: "preference",
      content: "User prefers tabs over spaces.",
    });

    assert.equal(second, first, "retry must return the existing id");
    const n = db
      .prepare("SELECT COUNT(*) AS n FROM memories WHERE user_id = 'u_dup'")
      .get() as { n: number };
    assert.equal(n.n, 1, "no duplicate row");

    const mem = getMemory(db, first);
    assert.equal(mem?.access_count, 1, "duplicate insert should touch the existing row");
  });

  it("content_hash: superseding frees the hash for re-learning", () => {
    const old = insertMemory(db, { user_id: "u_res", type: "preference", content: "Use Postgres." });
    updateMemoryStatus(db, old, "superseded");
    const again = insertMemory(db, { user_id: "u_res", type: "preference", content: "Use Postgres." });
    assert.notEqual(again, old, "same content may return as a NEW active memory after supersession");
  });

  it("content_hash: same content for different users does not collide", () => {
    const a = insertMemory(db, { user_id: "u_a", type: "fact", content: "Shared content." });
    const b = insertMemory(db, { user_id: "u_b", type: "fact", content: "Shared content." });
    assert.notEqual(a, b);
  });

  // ── v2: memory graph edges ──────────────────────────────────────────────────

  it("edges: round-trip in both directions, idempotent add", () => {
    const src = insertMemory(db, { user_id: "u_e", type: "preference", content: "New pref: SQLite." });
    const dst = insertMemory(db, { user_id: "u_e", type: "preference", content: "Old pref: Postgres." });

    const e1 = addEdge(db, src, dst, "supersedes");
    const e2 = addEdge(db, src, dst, "supersedes");
    assert.equal(e2, e1, "re-adding the same edge must return the existing id");

    const fromSrc = edgesFor(db, src);
    const fromDst = edgesFor(db, dst);
    assert.equal(fromSrc.length, 1);
    assert.deepEqual(
      { src_id: fromSrc[0].src_id, dst_id: fromSrc[0].dst_id, kind: fromSrc[0].kind },
      { src_id: src, dst_id: dst, kind: "supersedes" },
    );
    assert.deepEqual(fromDst, fromSrc, "edge must be visible from both endpoints");
  });

  it("edges: kind constraint rejects unknown kinds", () => {
    const a = insertMemory(db, { user_id: "u_e2", type: "fact", content: "A." });
    const b = insertMemory(db, { user_id: "u_e2", type: "fact", content: "B." });
    assert.throws(() =>
      db.prepare(
        "INSERT INTO memory_edges (id, src_id, dst_id, kind, created_at) VALUES ('x', ?, ?, 'bogus', 0)"
      ).run(a, b),
    );
  });

  // ── v2: belief_events + sleep_runs ──────────────────────────────────────────

  it("belief_events: insert and read back with reason check", () => {
    const m = insertMemory(db, { user_id: "u_b", type: "hypothesis", content: "User is deadline-driven." });
    const id = insertBeliefEvent(db, { memory_id: m, delta: 0.15, reason: "confirmed", turn_id: "t-1" });

    const row = db
      .prepare("SELECT * FROM belief_events WHERE id=?")
      .get(id) as { memory_id: string; delta: number; reason: string; turn_id: string };
    assert.equal(row.memory_id, m);
    assert.equal(row.delta, 0.15);
    assert.equal(row.reason, "confirmed");
    assert.equal(row.turn_id, "t-1");
  });

  it("sleep_runs: insert and read back counts", () => {
    const id = insertSleepRun(db, {
      user_id: "u_s",
      consolidated_n: 2,
      decayed_n: 5,
      inferred_n: 1,
      notes: "first dream",
    });
    const row = db
      .prepare("SELECT * FROM sleep_runs WHERE id=?")
      .get(id) as { consolidated_n: number; decayed_n: number; inferred_n: number; notes: string };
    assert.deepEqual(
      { c: row.consolidated_n, d: row.decayed_n, i: row.inferred_n },
      { c: 2, d: 5, i: 1 },
    );
  });

  // ── v2: hypothesis type + inferred source + session provenance ─────────────

  it("v2 columns: hypothesis/inferred/session_id round-trip", () => {
    const id = insertMemory(db, {
      user_id: "u_v2",
      type: "hypothesis",
      content: "User always cuts scope near deadlines.",
      source: "inferred",
      confidence: 0.4,
      session_id: "session-3",
    });
    const mem = getMemory(db, id);
    assert.ok(mem);
    assert.equal(mem.type, "hypothesis");
    assert.equal(mem.source, "inferred");
    assert.equal(mem.confidence, 0.4);
    assert.equal(mem.session_id, "session-3");
  });

  it("v2: setConfidence updates the stored value", () => {
    const id = insertMemory(db, { user_id: "u_c", type: "fact", content: "Confidence test." });
    setConfidence(db, id, 0.25);
    assert.equal(getMemory(db, id)?.confidence, 0.25);
  });

  it("v2: recentEpisodic returns newest-first, limited", () => {
    for (let i = 0; i < 5; i++) {
      insertEpisodic(db, { user_id: "u_ep", session_id: "s1", role: "user", content: `turn ${i}` });
    }
    const rows = recentEpisodic(db, "u_ep", 3);
    assert.equal(rows.length, 3);
    assert.equal(rows[0].content, "turn 4", "newest first");
  });

  it("v2: insertEvalRun stores metrics row", () => {
    const id = insertEvalRun(db, {
      arm: "engram",
      session_index: 3,
      task_accuracy: 0.8,
      tokens_in_context: 1400,
      metadata: { seed: 42 },
    });
    const row = db
      .prepare("SELECT * FROM eval_runs WHERE id=?")
      .get(id) as { arm: string; task_accuracy: number; metadata: string };
    assert.equal(row.arm, "engram");
    assert.equal(row.task_accuracy, 0.8);
    assert.equal(JSON.parse(row.metadata).seed, 42);
  });
});
