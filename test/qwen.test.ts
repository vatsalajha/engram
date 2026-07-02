/**
 * test/qwen.test.ts — unit tests for src/llm/qwen.ts with the HTTP layer mocked.
 *
 * The qwen client is constructed with `fetch: (url, init) => globalThis.fetch(...)`,
 * so stubbing globalThis.fetch intercepts every request without network access.
 *
 * Covers (per SPEC / Prompt 1):
 *   - fence-stripping (direct + through chatJSON)
 *   - chatJSON retry-on-invalid-JSON (error fed back, throws after 2nd failure)
 *   - embed batch chunking (≤10 per request, order preserved)
 *   - usage accumulation (per-model buckets + aggregate totals)
 */

import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";

import {
  chat,
  chatJSON,
  embed,
  getUsageStats,
  resetUsage,
  stripFences,
  Models,
} from "../src/llm/qwen.js";

// ─── fetch mock harness ───────────────────────────────────────────────────────

type RecordedCall = { url: string; body: any };

const realFetch = globalThis.fetch;
let recorded: RecordedCall[] = [];
let responders: Array<() => Response> = [];

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function chatPayload(content: string, usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number }) {
  return {
    id: "chatcmpl-test",
    object: "chat.completion",
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
    usage: usage ?? { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };
}

function embedPayload(count: number, offset: number) {
  return {
    object: "list",
    data: Array.from({ length: count }, (_, i) => ({
      object: "embedding",
      index: i,
      // Encode global position in the first component so order is verifiable.
      embedding: [offset + i, 0, 0, 0],
    })),
    usage: { prompt_tokens: count, total_tokens: count },
  };
}

beforeEach(() => {
  recorded = [];
  responders = [];
  resetUsage();
  globalThis.fetch = (async (url: any, init?: any) => {
    recorded.push({
      url: String(url),
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    });
    const next = responders.shift();
    if (!next) throw new Error(`fetch mock: no responder queued for call #${recorded.length}`);
    return next();
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

// ─── stripFences ──────────────────────────────────────────────────────────────

describe("stripFences (unit)", () => {
  it("strips ```json fences", () => {
    assert.equal(stripFences('```json\n{"a":1}\n```'), '{"a":1}');
  });

  it("strips bare ``` fences", () => {
    assert.equal(stripFences('```\n{"a":1}\n```'), '{"a":1}');
  });

  it("leaves clean JSON untouched", () => {
    assert.equal(stripFences('{"a":1}'), '{"a":1}');
  });
});

// ─── chatJSON ─────────────────────────────────────────────────────────────────

describe("chatJSON (mocked HTTP)", () => {
  const schema = z.object({ name: z.string(), count: z.number() });
  const opts = {
    system: "extract",
    messages: [{ role: "user" as const, content: "hi" }],
    model: Models.plus,
  };

  it("parses fenced JSON in one call", async () => {
    responders = [() => jsonResponse(chatPayload('```json\n{"name":"engram","count":2}\n```'))];
    const out = await chatJSON(opts, schema);
    assert.deepEqual(out, { name: "engram", count: 2 });
    assert.equal(recorded.length, 1);
  });

  it("retries once on invalid JSON, feeding the error back", async () => {
    responders = [
      () => jsonResponse(chatPayload("definitely not json")),
      () => jsonResponse(chatPayload('{"name":"engram","count":7}')),
    ];
    const out = await chatJSON(opts, schema);
    assert.deepEqual(out, { name: "engram", count: 7 });
    assert.equal(recorded.length, 2);
    const retrySystem = recorded[1].body.messages[0].content as string;
    assert.match(retrySystem, /failed validation/i, "retry should feed the error back in the system prompt");
  });

  it("retries once on Zod mismatch, then throws after the 2nd failure", async () => {
    responders = [
      () => jsonResponse(chatPayload('{"name":"engram","count":"not-a-number"}')),
      () => jsonResponse(chatPayload('{"still":"wrong"}')),
    ];
    await assert.rejects(() => chatJSON(opts, schema), /chatJSON failed after 2 attempts/);
    assert.equal(recorded.length, 2);
  });
});

// ─── embed batching ───────────────────────────────────────────────────────────

describe("embed (mocked HTTP)", () => {
  it("chunks into batches of ≤10 and preserves input order", async () => {
    const texts = Array.from({ length: 23 }, (_, i) => `text-${i}`);
    responders = [
      () => jsonResponse(embedPayload(10, 0)),
      () => jsonResponse(embedPayload(10, 10)),
      () => jsonResponse(embedPayload(3, 20)),
    ];
    const vectors = await embed(texts, 4);

    assert.equal(recorded.length, 3, "23 texts → 3 requests");
    assert.deepEqual(
      recorded.map((c) => c.body.input.length),
      [10, 10, 3],
      "batch sizes must be ≤10",
    );
    assert.equal(vectors.length, 23);
    // First component encodes global position — must be 0..22 in order.
    assert.deepEqual(vectors.map((v) => v[0]), Array.from({ length: 23 }, (_, i) => i));
  });

  it("returns [] for empty input without any HTTP call", async () => {
    const vectors = await embed([]);
    assert.deepEqual(vectors, []);
    assert.equal(recorded.length, 0);
  });
});

// ─── usage accounting ─────────────────────────────────────────────────────────

describe("usage accounting (mocked HTTP)", () => {
  it("accumulates per model and aggregates totals", async () => {
    responders = [
      () => jsonResponse(chatPayload("a", { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 })),
      () => jsonResponse(chatPayload("b", { prompt_tokens: 40, completion_tokens: 10, total_tokens: 50 })),
      () => jsonResponse(embedPayload(2, 0)),
    ];

    await chat({ system: "s", messages: [{ role: "user", content: "1" }], model: Models.plus });
    await chat({ system: "s", messages: [{ role: "user", content: "2" }], model: Models.flash });
    await embed(["x", "y"], 4);

    const stats = getUsageStats();
    assert.equal(stats.byModel[Models.plus].totalTokens, 120);
    assert.equal(stats.byModel[Models.plus].calls, 1);
    assert.equal(stats.byModel[Models.flash].totalTokens, 50);
    assert.equal(stats.byModel[Models.embedding].promptTokens, 2);
    assert.equal(stats.byModel[Models.embedding].completionTokens, 0);
    // Aggregate = sum of buckets
    assert.equal(stats.totalTokens, 120 + 50 + 2);
    assert.equal(stats.calls, 3);
  });

  it("resetUsage clears all buckets", async () => {
    responders = [() => jsonResponse(chatPayload("a"))];
    await chat({ system: "s", messages: [{ role: "user", content: "1" }] });
    assert.ok(getUsageStats().totalTokens > 0);
    resetUsage();
    const stats = getUsageStats();
    assert.equal(stats.totalTokens, 0);
    assert.equal(stats.calls, 0);
    assert.deepEqual(stats.byModel, {});
  });
});
