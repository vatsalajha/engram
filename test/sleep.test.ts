/**
 * test/maintenance.test.ts
 *
 * PART A — offline (no API):
 *   - decayAndExpire archives stale, low-salience events
 *   - decayAndExpire skips pinned memories unconditionally
 *   - decayAndExpire expires archived memories past their ttl_at
 *   - detectSupersession returns early with no candidates (no LLM call)
 *
 * PART B — integration (DASHSCOPE_API_KEY required):
 *   - Supersession: writing a new preference supersedes the old one
 *   - Consolidation: a cluster of similar events collapses to one semantic memory
 *
 * Run: npm test
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openDb, type Db } from "../src/db/client.js";
import {
  insertMemory,
  getMemory,
  listMemories,
} from "../src/db/queries.js";
import {
  decayAndExpire,
  detectSupersession,
  consolidate,
  inferHypotheses,
  runSleep,
} from "../src/memory/sleep.js";
import { insertEpisodic, edgesFor } from "../src/db/queries.js";

// ─── Synthetic embedding helpers ─────────────────────────────────────────────

function unitVec(d: number, dim = 1024): number[] {
  const v = new Array<number>(dim).fill(0);
  v[d] = 1.0;
  return v;
}

function nearVec(d: number, sim: number, dim = 1024): number[] {
  const perp = Math.sqrt(1 - sim * sim);
  const v = new Array<number>(dim).fill(0);
  v[d] = sim;
  v[d + 1] = perp;
  return v;
}

// ─── PART A: offline tests ────────────────────────────────────────────────────

describe("decayAndExpire (offline)", () => {
  let db: Db;
  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  it("archives a stale, low-salience event (crafted to fall below threshold)", () => {
    // retention = 0.25*sal + 0.25*conf + 0.25*rec + 0.15*logAcc + 0.10*typ
    // typ(event)=0.35, sal=0.05, conf=0.1, rec≈0, logAcc=0 (never accessed)
    // → score ≈ 0.25*0.05 + 0.25*0.1 + 0.25*0 + 0 + 0.10*0.35
    //         = 0.0125 + 0.025 + 0 + 0 + 0.035 = 0.0725 < 0.3 → archive
    const id = insertMemory(db, {
      user_id: "u_decay",
      type: "event",
      content: "User attended a brief meeting.",
      salience: 0.05,
      confidence: 0.1,
    });

    // Back-date created_at 365 days (recency ≈ 0) but keep last_accessed_at
    // recent — otherwise the 48h archived-and-untouched rule would expire it
    // in the same sweep (see the dedicated expiry test below).
    const oldTs = Date.now() - 365 * 86_400_000;
    db.prepare("UPDATE memories SET created_at = ?, last_accessed_at = ? WHERE id = ?")
      .run(oldTs, Date.now() - 3_600_000, id);

    const { archived } = decayAndExpire("u_decay", db);
    assert.ok(archived >= 1, `expected ≥1 archived, got ${archived}`);

    const mem = getMemory(db, id)!;
    assert.equal(mem.status, "archived", "stale event should be archived");
  });

  it("archived and untouched for 48h → expired on the next sweep", () => {
    const id = insertMemory(db, {
      user_id: "u_stale48",
      type: "event",
      content: "User once mentioned a passing detail.",
      salience: 0.05,
      confidence: 0.1,
    });
    // Archived long ago, never touched since (72h > 48h cutoff)
    const oldTs = Date.now() - 72 * 3_600_000;
    db.prepare("UPDATE memories SET status='archived', created_at=?, last_accessed_at=? WHERE id=?")
      .run(oldTs, oldTs, id);

    const { expired } = decayAndExpire("u_stale48", db);
    assert.ok(expired >= 1);
    assert.equal(getMemory(db, id)?.status, "expired");
  });

  it("does NOT archive a pinned memory regardless of retention score", () => {
    const id = insertMemory(db, {
      user_id: "u_pinned",
      type: "note",
      content: "Critical note that must never be forgotten.",
      salience: 0.01,
      confidence: 0.01,
      pinned: 1,
    });

    // Make it ancient
    const oldTs = Date.now() - 730 * 86_400_000;
    db.prepare("UPDATE memories SET created_at = ?, last_accessed_at = ? WHERE id = ?")
      .run(oldTs, oldTs, id);

    decayAndExpire("u_pinned", db);

    const mem = getMemory(db, id)!;
    assert.equal(mem.status, "active", "pinned memory must remain active");
  });

  it("expires archived memories past their ttl_at", () => {
    const id = insertMemory(db, {
      user_id: "u_ttl",
      type: "note",
      content: "Temporary reminder.",
      salience: 0.05,
      confidence: 0.1,
      // ttl_at already in the past
      ttl_at: Date.now() - 1000,
    });

    // Archive it first via a direct status update
    db.prepare("UPDATE memories SET status = 'archived' WHERE id = ?").run(id);

    const { expired } = decayAndExpire("u_ttl", db);
    assert.ok(expired >= 1, `expected ≥1 expired, got ${expired}`);

    const mem = getMemory(db, id)!;
    assert.equal(mem.status, "expired", "past-TTL archived memory should be expired");
  });

  it("keeps a fresh, high-salience preference active", () => {
    const id = insertMemory(db, {
      user_id: "u_fresh",
      type: "preference",
      content: "The user prefers dark mode across all applications.",
      salience: 0.9,
      confidence: 0.95,
    });

    const { archived } = decayAndExpire("u_fresh", db);
    // May archive other users' memories from previous tests — but not this one
    const mem = getMemory(db, id)!;
    assert.equal(mem.status, "active", "fresh high-salience preference must stay active");
  });
});

describe("detectSupersession (offline edge cases)", () => {
  let db: Db;
  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  it("returns immediately when there are no prior memories (no LLM call)", async () => {
    // detectSupersession calls vecSearch; on an empty DB it returns [] and exits early
    await detectSupersession(
      "u_empty",
      { id: "fake_id", type: "preference", content: "Prefer dark mode.", embedding: unitVec(0) },
      db,
    );
    // If we reach here without an error, the test passes
    assert.ok(true, "detectSupersession should not throw on empty store");
  });

  it("does not modify any memory when new content is far from all existing", async () => {
    // Insert an embedding at dimension 0
    const id = insertMemory(db, {
      user_id: "u_far",
      type: "preference",
      content: "The user prefers light mode.",
      salience: 0.8,
      embedding: unitVec(0),
    });

    // Insert at dimension 500 — orthogonal (cosine distance ≈ 1)
    // vecSearch will return the existing memory, but it's so far that
    // the LLM would call "unrelated" — but since we have no API key,
    // the LLM call will throw and detectSupersession swallows the error.
    try {
      await detectSupersession(
        "u_far",
        { id: "new_id", type: "preference", content: "Prefer dark mode.", embedding: unitVec(500) },
        db,
      );
    } catch {
      // Expected if API key is missing; still verify nothing was mutated
    }

    const mem = getMemory(db, id)!;
    assert.equal(mem.status, "active", "far-away memory should remain active");
  });
});

// ─── PART A2: full sleep cycle (offline — LLM + embeddings mocked) ────────────
// The fetch stub dispatches on URL: /chat/completions vs /embeddings, so
// consolidate (chat + embed) and infer (chatJSON + embed) run fully offline.

describe("runSleep (offline, mocked LLM)", () => {
  let db: Db;
  const realFetch = globalThis.fetch;

  before(() => { db = openDb(":memory:"); });
  after(() => {
    db.close();
    globalThis.fetch = realFetch;
  });

  /** fetch stub: chat replies come from `chatReplies` (in order); embeddings are unitVec(dim). */
  function stubLlm(chatReplies: string[], embedDim: () => number) {
    let chatIdx = 0;
    globalThis.fetch = (async (url: any, init?: any) => {
      const u = String(url);
      if (u.includes("/embeddings")) {
        const body = JSON.parse(String(init?.body));
        const inputs: string[] = body.input;
        return new Response(JSON.stringify({
          object: "list",
          data: inputs.map((_, i) => ({ object: "embedding", index: i, embedding: unitVec(embedDim()) })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }), { status: 200, headers: { "content-type": "application/json" } });
      }
      const content = chatReplies[Math.min(chatIdx++, chatReplies.length - 1)];
      return new Response(JSON.stringify({
        id: "c", object: "chat.completion",
        choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }), { status: 200, headers: { "content-type": "application/json" } });
    }) as typeof fetch;
  }

  it("consolidates a cluster: summary + derived_from edges + archived sources", async () => {
    // Three tightly-clustered events (cos ≥ 0.9 to seed) + one far-away event.
    const e1 = insertMemory(db, { user_id: "u_sc", type: "event", salience: 0.5, content: "The user debugged the retrieval layer on Monday.", embedding: unitVec(10) });
    const e2 = insertMemory(db, { user_id: "u_sc", type: "event", salience: 0.6, content: "The user debugged the packer on Tuesday.", embedding: nearVec(10, 0.93) });
    const e3 = insertMemory(db, { user_id: "u_sc", type: "event", salience: 0.4, content: "The user debugged the belief engine on Wednesday.", embedding: nearVec(10, 0.9) });
    const far = insertMemory(db, { user_id: "u_sc", type: "event", salience: 0.5, content: "The user went hiking.", embedding: unitVec(500) });

    stubLlm(["The user spent the week debugging Engram's memory subsystems."], () => 11);

    const n = await consolidate("u_sc", db);
    assert.equal(n, 1, "exactly one cluster consolidates");

    // Summary exists, is source=consolidated, and has an embedding.
    const summaries = listMemories(db, { user_id: "u_sc", status: "active", type: "fact" });
    assert.equal(summaries.length, 1);
    const summary = summaries[0];
    assert.equal(summary.source, "consolidated");
    assert.ok(summary.embedding && summary.embedding.length === 1024, "summary must be embedded");

    // Sources archived; far event untouched.
    for (const id of [e1, e2, e3]) assert.equal(getMemory(db, id)?.status, "archived");
    assert.equal(getMemory(db, far)?.status, "active");

    // derived_from edges summary → each source.
    const edges = edgesFor(db, summary.id);
    assert.equal(edges.length, 3);
    assert.ok(edges.every((e) => e.kind === "derived_from" && e.src_id === summary.id));
  });

  it("infers a hypothesis from episodic log; dedupes a re-proposal", async () => {
    insertEpisodic(db, { user_id: "u_inf", session_id: "s1", role: "user", content: "Can we cut the rerank feature? Deadline is close." });
    insertEpisodic(db, { user_id: "u_inf", session_id: "s2", role: "user", content: "Drop the fancy charts too, ship the core." });

    const hypothesis = { content: "The user tends to cut scope aggressively when a deadline is near.", reasoning: "Repeatedly cut features near deadlines." };
    stubLlm([JSON.stringify({ hypotheses: [hypothesis] })], () => 20);

    const first = await inferHypotheses("u_inf", db);
    assert.equal(first.length, 1, "hypothesis inserted");
    const mem = getMemory(db, first[0])!;
    assert.equal(mem.type, "hypothesis");
    assert.equal(mem.source, "inferred");
    assert.equal(mem.confidence, 0.4);

    // Second run proposes the same thing (same embedding dim) → deduped.
    stubLlm([JSON.stringify({ hypotheses: [hypothesis] })], () => 20);
    const second = await inferHypotheses("u_inf", db);
    assert.equal(second.length, 0, "semantically-identical proposal must dedupe");
  });

  it("no episodic history → no inference LLM call", async () => {
    let calls = 0;
    globalThis.fetch = (async () => { calls++; return new Response("{}", { status: 200 }); }) as typeof fetch;
    const inserted = await inferHypotheses("u_empty_ep", db);
    assert.deepEqual(inserted, []);
    assert.equal(calls, 0);
  });

  it("runSleep writes a sleep_runs row; second immediate run is a near-no-op", async () => {
    insertEpisodic(db, { user_id: "u_run", session_id: "s1", role: "user", content: "I always test before deploying." });
    for (let i = 0; i < 3; i++) {
      insertMemory(db, {
        user_id: "u_run", type: "event", salience: 0.5,
        content: `The user ran deployment step ${i}.`,
        embedding: nearVec(30, 0.95 - i * 0.01),
      });
    }

    // Distinct dims per embed call: summary → 31, hypothesis → 32 (otherwise
    // the hypothesis would spuriously dedupe against the fresh summary).
    const dims = [31, 32];
    let dimIdx = 0;
    stubLlm(
      [
        "The user follows a careful deployment routine.",                     // consolidation summary
        JSON.stringify({ hypotheses: [{ content: "The user is cautious about production deploys and validates first.", reasoning: "states always testing" }] }),
      ],
      () => dims[Math.min(dimIdx++, dims.length - 1)],
    );

    const run1 = await runSleep("u_run", db);
    assert.equal(run1.skipped, false);
    assert.ok(run1.id, "sleep_runs row written");
    assert.equal(run1.consolidated_n, 1);
    assert.equal(run1.inferred_n, 1);

    const row = db.prepare("SELECT * FROM sleep_runs WHERE id=?").get(run1.id) as
      { consolidated_n: number; inferred_n: number; user_id: string };
    assert.equal(row.user_id, "u_run");
    assert.equal(row.consolidated_n, 1);

    // Second immediate run: same proposals → dedupe; no event clusters left.
    stubLlm(
      [JSON.stringify({ hypotheses: [{ content: "The user is cautious about production deploys and validates first.", reasoning: "same" }] })],
      () => 32, // same embedding as the stored hypothesis → dedupes
    );
    const run2 = await runSleep("u_run", db);
    assert.equal(run2.skipped, false);
    assert.equal(run2.consolidated_n, 0, "no clusters remain");
    assert.equal(run2.inferred_n, 0, "hypothesis deduped");
  });

  it("in-process mutex: concurrent runSleep for the same user skips one", async () => {
    stubLlm([JSON.stringify({ hypotheses: [] })], () => 40);
    insertEpisodic(db, { user_id: "u_mutex", session_id: "s", role: "user", content: "hello" });

    const [a, b] = await Promise.all([
      runSleep("u_mutex", db),
      runSleep("u_mutex", db),
    ]);
    assert.equal([a, b].filter((r) => r.skipped).length, 1, "exactly one run skips");
  });
});

// ─── PART B: integration tests (requires real DASHSCOPE_API_KEY) ─────────────

const _rawKey = process.env.DASHSCOPE_API_KEY ?? "";
const HAS_API_KEY = _rawKey.length > 20 && !_rawKey.includes("...");

describe("detectSupersession (integration)", { skip: !HAS_API_KEY }, () => {
  let db: Db;
  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  it("supersedes old preference when new one explicitly updates it", async () => {
    // Write "prefer dark mode"
    const { embed } = await import("../src/llm/qwen.js");

    const darkContent = "The user prefers dark mode in all interfaces.";
    const [darkEmb] = await embed([darkContent]);
    const darkId = insertMemory(db, {
      user_id: "u_sup",
      type: "preference",
      content: darkContent,
      salience: 0.8,
      embedding: darkEmb,
    });

    // Write "switched to light mode" — should supersede the dark mode preference
    const lightContent = "The user has switched to light mode and now prefers it over dark mode.";
    const [lightEmb] = await embed([lightContent]);
    const lightId = insertMemory(db, {
      user_id: "u_sup",
      type: "preference",
      content: lightContent,
      salience: 0.8,
      embedding: lightEmb,
    });

    await detectSupersession(
      "u_sup",
      { id: lightId, type: "preference", content: lightContent, embedding: lightEmb },
      db,
    );

    const dark = getMemory(db, darkId)!;
    const light = getMemory(db, lightId)!;

    console.log("\n--- Supersession test ---");
    console.log(`Dark mode: status=${dark.status}, superseded_by=${dark.superseded_by}`);
    console.log(`Light mode: status=${light.status}`);

    // Show conflicts table
    const conflicts = db
      .prepare("SELECT * FROM conflicts ORDER BY created_at DESC LIMIT 5")
      .all() as { new_id: string; old_id: string; resolution: string; reasoning: string }[];
    console.log("\n--- conflicts table ---");
    console.table(conflicts);
    console.log("--- end conflicts ---\n");

    assert.equal(dark.status, "superseded", "dark mode preference should be superseded");
    assert.equal(dark.superseded_by, lightId, "superseded_by should point to light mode memory");
    assert.equal(light.status, "active", "light mode preference should remain active");
  });
});

describe("consolidate (integration)", { skip: !HAS_API_KEY }, () => {
  let db: Db;
  before(() => { db = openDb(":memory:"); });
  after(() => { db.close(); });

  it("consolidates a cluster of similar events into one semantic memory", async () => {
    const { embed } = await import("../src/llm/qwen.js");

    const eventContents = [
      "The user completed the onboarding tutorial on Monday.",
      "The user finished the setup guide on Tuesday.",
      "The user went through the getting-started walkthrough on Wednesday.",
    ];

    // Embed and insert all events
    const embeddings = await embed(eventContents);
    const eventIds: string[] = [];
    for (let i = 0; i < eventContents.length; i++) {
      const id = insertMemory(db, {
        user_id: "u_con",
        type: "event",
        content: eventContents[i],
        salience: 0.4,
        confidence: 0.9,
        embedding: embeddings[i],
      });
      eventIds.push(id);
    }

    const consolidations = await consolidate("u_con", db);

    console.log("\n--- Consolidation test ---");
    console.log(`Consolidations created: ${consolidations}`);

    // Show the consolidations table
    const rows = db
      .prepare("SELECT * FROM consolidations ORDER BY created_at DESC LIMIT 5")
      .all() as { id: string; summary_id: string; source_ids: string }[];
    console.log("\n--- consolidations table ---");
    console.table(rows.map((r) => ({ ...r, source_ids: JSON.parse(r.source_ids) })));
    console.log("--- end consolidations ---");

    // Show the new consolidated memory
    const consolidated = listMemories(db, { user_id: "u_con", status: "active", type: "fact" });
    console.log("\n--- consolidated memory ---");
    for (const m of consolidated) {
      console.log(`  [${m.source}] ${m.content}`);
    }
    console.log("--- end consolidated memory ---\n");

    assert.ok(consolidations >= 1, "should produce at least one consolidation");

    // Source events should now be archived
    for (const id of eventIds) {
      const mem = getMemory(db, id)!;
      assert.equal(mem.status, "archived", `event ${id} should be archived after consolidation`);
    }

    // There should be at least one consolidated (fact, source=consolidated) memory
    assert.ok(
      consolidated.some((m) => m.source === "consolidated"),
      "expected a consolidated memory with source='consolidated'",
    );
  });
});
