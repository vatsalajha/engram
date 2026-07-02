/**
 * test/api.test.ts
 *
 * HTTP API tests — fully offline. The Hono app is imported with
 * ENGRAM_DB_PATH=:memory: and every LLM/embedding call is served by a
 * dispatching fetch stub:
 *   /embeddings                          → deterministic unit vectors by text
 *   /chat/completions (stream:true)      → SSE chunk stream (the /act answer)
 *   /chat/completions (extraction)       → one typed memory from the turn
 *   /chat/completions (judge/supersede)  → neutral / unrelated verdicts
 *
 * Covers: /act round-trip persistence → recallable on next /act; /feedback
 * mutating confidence; 429 rate limit; idempotent replay with cached result.
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";

// Environment MUST be set before the app (and config) load.
process.env.ENGRAM_DB_PATH = ":memory:";
process.env.ENGRAM_SCHEDULER = "off";
process.env.ENGRAM_RATE_LIMIT_MAX = "5";
process.env.ENGRAM_RERANK_ENABLED = "false";
// Empty string (NOT delete): dotenv would re-add REDIS_URL from .env, and an
// unreachable Redis keeps ioredis reconnecting forever — the test process
// would never exit. "" is falsy → the server runs with in-process fallbacks.
process.env.REDIS_URL = "";

const { default: app } = await import("../src/api/server.js");
const { getDb } = await import("../src/db/client.js");
const { insertMemory, getMemory, listMemories } = await import("../src/db/queries.js");
const { drainWrites } = await import("../src/agent/loop.js");

// ─── fetch stub ───────────────────────────────────────────────────────────────

const realFetch = globalThis.fetch;

/** Deterministic embedding: hash the text onto a unit-vector dimension. */
function embeddingFor(text: string): number[] {
  let h = 0;
  for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) | 0;
  const v = new Array<number>(1024).fill(0);
  v[Math.abs(h) % 1024] = 1;
  return v;
}

function sseBody(tokens: string[]): string {
  const chunks = tokens.map((t) =>
    `data: ${JSON.stringify({ id: "c", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: t } }] })}\n\n`,
  );
  chunks.push(
    `data: ${JSON.stringify({ id: "c", object: "chat.completion.chunk", choices: [], usage: { prompt_tokens: 50, completion_tokens: 10, total_tokens: 60 } })}\n\n`,
  );
  chunks.push("data: [DONE]\n\n");
  return chunks.join("");
}

function jsonChat(content: string): Response {
  return new Response(
    JSON.stringify({
      id: "c", object: "chat.completion",
      choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

/** What the extraction call should return, settable per test. */
let extractionMemories: object[] = [];
/** Captured system prompts of streamed (answer) calls, for context assertions. */
const streamedSystemPrompts: string[] = [];

function installFetchStub() {
  globalThis.fetch = (async (url: any, init?: any) => {
    const u = String(url);
    if (u.includes("/embeddings")) {
      const body = JSON.parse(String(init?.body));
      const inputs: string[] = body.input;
      return new Response(
        JSON.stringify({
          object: "list",
          data: inputs.map((t, i) => ({ object: "embedding", index: i, embedding: embeddingFor(t) })),
          usage: { prompt_tokens: inputs.length, total_tokens: inputs.length },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }

    const body = JSON.parse(String(init?.body));
    if (body.stream === true) {
      streamedSystemPrompts.push(body.messages[0].content as string);
      return new Response(sseBody(["Since ", "you told me, ", "your project is Artmoji."]), {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    }

    const system: string = body.messages[0].content;
    if (system.includes("memory extraction engine")) {
      return jsonChat(JSON.stringify({ memories: extractionMemories }));
    }
    if (system.includes("audit an agent's memory")) {
      // implicit judge — everything neutral (belief effects tested in belief.test)
      const userMsg: string = body.messages[1].content;
      const count = (userMsg.match(/^\d+\. /gm) ?? []).length;
      return jsonChat(JSON.stringify({
        verdicts: Array.from({ length: count }, (_, i) => ({
          index: i + 1, verdict: "neutral", reasoning: "n/a",
        })),
      }));
    }
    if (system.includes("compare a NEW memory")) {
      const userMsg: string = body.messages[1].content;
      const count = (userMsg.match(/^\d+\. /gm) ?? []).length;
      return jsonChat(JSON.stringify({
        verdicts: Array.from({ length: count }, (_, i) => ({
          old_index: i + 1, action: "unrelated", reasoning: "n/a",
        })),
      }));
    }
    // Rerank / compress / anything else: harmless default
    return jsonChat(JSON.stringify({ scores: [] }));
  }) as typeof fetch;
}

// ─── SSE response parsing ─────────────────────────────────────────────────────

interface SseEvent { event: string; data: string }

function parseSse(text: string): SseEvent[] {
  const events: SseEvent[] = [];
  for (const block of text.split("\n\n")) {
    const ev = /^event: (.+)$/m.exec(block)?.[1];
    const data = /^data: (.*)$/m.exec(block)?.[1];
    if (ev && data !== undefined) events.push({ event: ev, data });
  }
  return events;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe("HTTP API (offline, mocked LLM)", () => {
  before(() => installFetchStub());
  after(() => { globalThis.fetch = realFetch; });

  it("GET /health responds ok", async () => {
    const res = await app.request("/health");
    assert.equal(res.status, 200);
    const body = await res.json() as { ok: boolean };
    assert.equal(body.ok, true);
  });

  it("POST /act: turn 1 persists a memory; turn 2 recalls it into context", async () => {
    // ── Turn 1: teach it something ──────────────────────────────────────────
    extractionMemories = [{
      type: "fact",
      content: "The user's side project is called Artmoji.",
      salience: 0.8, confidence: 0.95, tags: ["project"],
    }];

    const res1 = await app.request("/act", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: "api_u1", sessionId: "s1", input: "My side project is called Artmoji." }),
    });
    assert.equal(res1.status, 200);
    const events1 = parseSse(await res1.text());
    assert.ok(events1.some((e) => e.event === "token"), "tokens streamed");
    const done1 = events1.find((e) => e.event === "done");
    assert.ok(done1, "final done event present");
    const doneData = JSON.parse(done1.data) as { turnId: string; manifest: unknown[]; usage: { totalTokens: number } };
    assert.ok(doneData.turnId.length === 26, "done carries a ULID turnId");
    assert.ok(doneData.usage.totalTokens > 0, "done carries usage");

    await drainWrites(); // post-response queue: episodic + extract + write + judge

    const written = listMemories(getDb(), { user_id: "api_u1", status: "active" });
    assert.equal(written.length, 1, "memory persisted after turn 1");
    assert.match(written[0].content, /Artmoji/);
    assert.equal(written[0].session_id, "s1", "session provenance recorded");

    // ── Turn 2: same topic → memory must be recalled into the context ───────
    extractionMemories = []; // nothing new to extract this turn
    const res2 = await app.request("/act", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: "api_u1", sessionId: "s2", input: "What is the user's side project called Artmoji about?" }),
    });
    assert.equal(res2.status, 200);
    const events2 = parseSse(await res2.text());
    const manifest = JSON.parse(events2.find((e) => e.event === "manifest")!.data) as { id: string }[];
    assert.equal(manifest.length, 1, "turn-1 memory appears in turn-2 manifest");
    assert.equal(manifest[0].id, written[0].id);

    const lastPrompt = streamedSystemPrompts.at(-1)!;
    assert.ok(lastPrompt.includes("What you know about this user"), "context block injected");
    assert.ok(lastPrompt.includes("Artmoji"), "memory content reached the model");

    await drainWrites();
    // Episodic log captured both sides of both turns
    const episodic = getDb().prepare(
      "SELECT role, COUNT(*) n FROM episodic_log WHERE user_id='api_u1' GROUP BY role",
    ).all() as { role: string; n: number }[];
    assert.deepEqual(
      Object.fromEntries(episodic.map((r) => [r.role, r.n])),
      { user: 2, assistant: 2 },
    );
  });

  it("POST /feedback mutates confidence and logs belief events", async () => {
    const id = insertMemory(getDb(), {
      user_id: "api_u2", type: "fact", content: "The user deploys on Fridays.", confidence: 0.5,
    });

    const res = await app.request("/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: "api_u2", turnId: "t-1", outcome: "contradicted", memoryIds: [id] }),
    });
    assert.equal(res.status, 200);
    const body = await res.json() as { applied: { confidence_after: number }[] };
    assert.equal(body.applied.length, 1);
    assert.ok(Math.abs(body.applied[0].confidence_after - 0.35) < 1e-9);
    assert.ok(Math.abs(getMemory(getDb(), id)!.confidence - 0.35) < 1e-9);
  });

  it("POST /feedback with Idempotency-Key: replay returns cached result, no double-apply", async () => {
    const id = insertMemory(getDb(), {
      user_id: "api_u3", type: "fact", content: "The user likes green tea.", confidence: 0.5,
    });

    const call = () => app.request("/feedback", {
      method: "POST",
      headers: { "content-type": "application/json", "Idempotency-Key": "fb-once" },
      body: JSON.stringify({ userId: "api_u3", outcome: "confirmed", memoryIds: [id] }),
    });

    const res1 = await call();
    assert.equal(res1.status, 200);
    const body1 = await res1.json();

    const res2 = await call();
    assert.equal(res2.status, 200);
    assert.equal(res2.headers.get("Idempotency-Replayed"), "true");
    const body2 = await res2.json();
    assert.deepEqual(body2, body1, "replay returns the original result");

    // Confidence moved exactly once: 0.5 + 0.15 = 0.65
    assert.ok(Math.abs(getMemory(getDb(), id)!.confidence - 0.65) < 1e-9);
  });

  it("GET /memories returns list + edges", async () => {
    const res = await app.request("/memories?user_id=api_u1");
    assert.equal(res.status, 200);
    const body = await res.json() as { memories: unknown[]; edges: unknown[]; count: number };
    assert.ok(body.count >= 1);
    assert.ok(Array.isArray(body.edges));
  });

  it("POST /sleep runs a cycle and returns the audit row", async () => {
    const res = await app.request("/sleep", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: "api_u1" }),
    });
    assert.equal(res.status, 200);
    const body = await res.json() as { id: string; skipped: boolean };
    assert.equal(body.skipped, false);
    assert.ok(body.id, "sleep_runs row id returned");
  });

  it("rate limit: 6th request in the window returns 429", async () => {
    // ENGRAM_RATE_LIMIT_MAX=5 for this test process; use a dedicated user.
    const call = () => app.request("/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: "api_rl", outcome: "confirmed", memoryIds: ["nonexistent"] }),
    });
    let last = 0;
    for (let i = 0; i < 6; i++) last = (await call()).status;
    assert.equal(last, 429);
  });

  it("Zod validation: malformed body → 400 with error JSON", async () => {
    const res = await app.request("/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userId: "x", outcome: "maybe", memoryIds: [] }),
    });
    assert.equal(res.status, 400);
    const body = await res.json() as { error: unknown };
    assert.ok(body.error, "consistent error JSON shape");
  });
});
