/**
 * test/belief.test.ts
 *
 * Belief engine (SPEC §4.4):
 *   applyFeedback — offline, pure DB: confidence steps, clamping, belief_events,
 *                   edges via anchor, floor enforcement (hypothesis vs fact).
 *   implicitJudge — offline with the qwen-plus call mocked via fetch stub.
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openDb, type Db } from "../src/db/client.js";
import { insertMemory, getMemory, edgesFor } from "../src/db/queries.js";
import { applyFeedback, implicitJudge, clamp01 } from "../src/memory/belief.js";
import { config } from "../src/config.js";

const ETA = config.ENGRAM_BELIEF_ETA;       // 0.15 default
const FLOOR = config.ENGRAM_CONFIDENCE_FLOOR; // 0.25 default

function seed(db: Db, over: Partial<Parameters<typeof insertMemory>[1]> = {}): string {
  return insertMemory(db, {
    user_id: "u_b",
    type: "fact",
    content: over.content ?? `Seed fact ${Math.random()}`,
    confidence: 0.5,
    ...over,
  } as Parameters<typeof insertMemory>[1]);
}

describe("applyFeedback (offline)", () => {
  let db: Db;
  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  it("clamp01 clamps both rails", () => {
    assert.equal(clamp01(-0.2), 0);
    assert.equal(clamp01(0.4), 0.4);
    assert.equal(clamp01(1.7), 1);
  });

  it("confirmation raises confidence by η and logs a belief event", () => {
    const id = seed(db, { content: "The user deploys before noon." });
    const [r] = applyFeedback(db, "u_b", "turn-1", "confirmed", [id]);

    assert.equal(r.confidence_before, 0.5);
    assert.ok(Math.abs(r.confidence_after - (0.5 + ETA)) < 1e-9);
    assert.equal(getMemory(db, id)?.confidence, r.confidence_after);

    const events = db.prepare("SELECT * FROM belief_events WHERE memory_id=?").all(id) as
      { delta: number; reason: string; turn_id: string }[];
    assert.equal(events.length, 1);
    assert.equal(events[0].reason, "confirmed");
    assert.ok(Math.abs(events[0].delta - ETA) < 1e-9);
    assert.equal(events[0].turn_id, "turn-1");
  });

  it("contradiction lowers confidence by η", () => {
    const id = seed(db, { content: "The user prefers Postgres." });
    const [r] = applyFeedback(db, "u_b", "turn-2", "contradicted", [id]);
    assert.ok(Math.abs(r.confidence_after - (0.5 - ETA)) < 1e-9);
  });

  it("clamping: confirmations cannot push confidence above 1", () => {
    const id = seed(db, { content: "Rock solid fact.", confidence: 0.95 });
    const [r] = applyFeedback(db, "u_b", "t", "confirmed", [id]);
    assert.equal(r.confidence_after, 1);
    // delta recorded is the ACTUAL change (0.05), not η
    const [ev] = db.prepare("SELECT delta FROM belief_events WHERE memory_id=?").all(id) as { delta: number }[];
    assert.ok(Math.abs(ev.delta - 0.05) < 1e-9);
  });

  it("repeated contradictions drive a hypothesis below the floor → archived", () => {
    const id = seed(db, {
      type: "hypothesis",
      content: "The user is a morning person.",
      confidence: 0.4,
      source: "inferred",
    });

    // 0.40 → 0.25 (not < floor) → 0.10 (< floor → archived)
    applyFeedback(db, "u_b", "t1", "contradicted", [id]);
    assert.equal(getMemory(db, id)?.status, "active", "0.25 is not below the floor yet");

    const [r2] = applyFeedback(db, "u_b", "t2", "contradicted", [id]);
    assert.equal(r2.floor_action, "archived");
    assert.equal(getMemory(db, id)?.status, "archived", "hypothesis below floor must die");

    // Further feedback on an archived memory is a no-op (scope guard).
    const r3 = applyFeedback(db, "u_b", "t3", "contradicted", [id]);
    assert.equal(r3.length, 0);
  });

  it("fact below the floor is flagged needs_review, NOT archived", () => {
    const id = seed(db, { type: "fact", content: "The user lives in Boston.", confidence: 0.3 });
    const [r] = applyFeedback(db, "u_b", "t", "contradicted", [id]); // 0.3 → 0.15 < 0.25
    assert.equal(r.floor_action, "needs_review");
    const mem = getMemory(db, id);
    assert.equal(mem?.status, "active", "facts are not auto-archived");
    assert.equal(mem?.needs_review, 1);
  });

  it("anchor memory produces supports/contradicts edges", () => {
    const anchor = seed(db, { content: "The user said they now use SQLite (turn anchor)." });
    const target = seed(db, { content: "The user values fast local iteration." });

    applyFeedback(db, "u_b", "t", "confirmed", [target], { anchorMemoryId: anchor });
    const edges = edgesFor(db, target);
    assert.equal(edges.length, 1);
    assert.deepEqual(
      { src: edges[0].src_id, dst: edges[0].dst_id, kind: edges[0].kind },
      { src: anchor, dst: target, kind: "supports" },
    );
  });

  it("user isolation: feedback cannot touch another user's memory", () => {
    const foreign = insertMemory(db, {
      user_id: "someone_else",
      type: "fact",
      content: "Foreign memory.",
      confidence: 0.5,
    });
    const results = applyFeedback(db, "u_b", "t", "contradicted", [foreign]);
    assert.equal(results.length, 0);
    assert.equal(getMemory(db, foreign)?.confidence, 0.5, "untouched");
  });
});

// ─── implicitJudge (offline, mocked qwen-plus) ────────────────────────────────

describe("implicitJudge (offline, mocked)", () => {
  let db: Db;
  const realFetch = globalThis.fetch;

  before(() => { db = openDb(":memory:"); });
  after(() => {
    db.close();
    globalThis.fetch = realFetch;
  });

  function mockVerdicts(verdicts: object[]) {
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({
          id: "c", object: "chat.completion",
          choices: [{
            index: 0,
            message: { role: "assistant", content: JSON.stringify({ verdicts }) },
            finish_reason: "stop",
          }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      )) as typeof fetch;
  }

  it("applies supported/contradicted, ignores neutral", async () => {
    const m1 = insertMemory(db, { user_id: "u_j", type: "preference", content: "The user prefers dark mode.", confidence: 0.6 });
    const m2 = insertMemory(db, { user_id: "u_j", type: "fact", content: "The user works at a startup.", confidence: 0.6 });
    const m3 = insertMemory(db, { user_id: "u_j", type: "note", content: "The user hikes on weekends.", confidence: 0.6 });

    mockVerdicts([
      { index: 1, verdict: "supported", reasoning: "User praised the dark theme." },
      { index: 2, verdict: "contradicted", reasoning: "User said they joined a big company." },
      { index: 3, verdict: "neutral", reasoning: "Not mentioned." },
    ]);

    const result = await implicitJudge(
      db, "u_j", "turn-9",
      [{ id: m1 }, { id: m2 }, { id: m3 }],
      { user: "Love the dark theme! By the way, I just joined BigCorp.", assistant: "Congrats!" },
    );

    assert.equal(result.judged, 3);
    assert.equal(result.applied.length, 2, "neutral is not applied");
    assert.ok(Math.abs(getMemory(db, m1)!.confidence - (0.6 + ETA)) < 1e-9);
    assert.ok(Math.abs(getMemory(db, m2)!.confidence - (0.6 - ETA)) < 1e-9);
    assert.equal(getMemory(db, m3)!.confidence, 0.6, "neutral memory untouched");

    const events = db.prepare(
      "SELECT memory_id, delta, reason FROM belief_events WHERE turn_id='turn-9' ORDER BY id",
    ).all() as { memory_id: string; delta: number; reason: string }[];
    assert.equal(events.length, 2);
  });

  it("LLM failure degrades gracefully to zero applied", async () => {
    globalThis.fetch = (async () => new Response("boom", { status: 500 })) as typeof fetch;
    const m = insertMemory(db, { user_id: "u_j2", type: "fact", content: "Some fact.", confidence: 0.6 });
    const result = await implicitJudge(db, "u_j2", "t", [{ id: m }], { user: "hello" });
    assert.deepEqual(result, { judged: 0, applied: [] });
    assert.equal(getMemory(db, m)?.confidence, 0.6);
  });

  it("empty manifest short-circuits without an LLM call", async () => {
    let called = 0;
    globalThis.fetch = (async () => { called++; return new Response("{}", { status: 200 }); }) as typeof fetch;
    const result = await implicitJudge(db, "u_j3", "t", [], { user: "hello" });
    assert.deepEqual(result, { judged: 0, applied: [] });
    assert.equal(called, 0);
  });
});
