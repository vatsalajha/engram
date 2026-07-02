/**
 * test/packer.test.ts
 *
 * PART A — offline: budget hard guarantee (property-style), reserved sub-budget,
 * pinned survival at tiny budgets, manifest integrity, session_born provenance.
 * PART A2 — compression path with the qwen-flash call mocked via fetch stub.
 *
 * Token counting: tiktoken o200k_base (real encoder, deterministic).
 */

import { describe, it, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import {
  packContext,
  countTokens,
  type ScoredCandidate,
} from "../src/memory/packer.js";
import type { MemoryType } from "../src/db/queries.js";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeCandidate(
  id: string,
  content: string,
  type: MemoryType,
  utilityScore: number,
  pinned = 0,
  session_id: string | null = null,
): ScoredCandidate {
  return {
    id, content, type, salience: 0.5, pinned,
    utility: pinned === 1 ? Infinity : utilityScore,
    session_id,
  };
}

/** Deterministic PRNG (mulberry32) so the property loop is reproducible. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const realFetch = globalThis.fetch;

/** Stub the LLM so any compression call returns a short fixed rewrite. */
function stubCompression(replyText = "Compressed decision-relevant fact.") {
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion",
        choices: [{ index: 0, message: { role: "assistant", content: replyText }, finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    )) as typeof fetch;
}

// ─── PART A: offline tests ────────────────────────────────────────────────────

describe("packContext (offline)", () => {
  after(() => { globalThis.fetch = realFetch; });
  beforeEach(() => { stubCompression(); }); // no test in this suite may hit the network

  it("countTokens: o200k_base — empty is 0, monotone in content length", () => {
    assert.equal(countTokens(""), 0);
    const short = countTokens("hello world");
    const long  = countTokens("hello world, this is a much longer sentence about the user.");
    assert.ok(short > 0);
    assert.ok(long > short, "longer text must cost more tokens");
  });

  it("empty candidates: returns empty text, zero tokens, empty manifest", async () => {
    const result = await packContext([], 500);
    assert.equal(result.text, "");
    assert.equal(result.totalTokens, 0);
    assert.deepEqual(result.manifest, []);
  });

  it("PROPERTY: budget never exceeded across 60 randomized inputs", async () => {
    const rand = mulberry32(42);
    const types: MemoryType[] = ["fact", "preference", "decision", "event", "skill", "hypothesis", "note"];
    const words = ["user", "prefers", "project", "database", "deadline", "typescript",
      "deploys", "morning", "sqlite", "always", "never", "scope", "dark", "tests"];

    for (let round = 0; round < 60; round++) {
      const n = 1 + Math.floor(rand() * 12);
      const candidates: ScoredCandidate[] = Array.from({ length: n }, (_, i) => {
        const len = 4 + Math.floor(rand() * 60);
        const content = Array.from({ length: len }, () => words[Math.floor(rand() * words.length)]).join(" ");
        return makeCandidate(
          `r${round}-m${i}`,
          `The ${content}.`,
          types[Math.floor(rand() * types.length)],
          rand(),
          rand() < 0.15 ? 1 : 0,
        );
      });
      const budget = 40 + Math.floor(rand() * 500);

      const result = await packContext(candidates, budget);

      assert.ok(
        result.totalTokens <= budget,
        `round ${round}: totalTokens ${result.totalTokens} > budget ${budget}`,
      );
      const manifestSum = result.manifest.reduce((s, m) => s + m.tokens, 0);
      assert.equal(result.totalTokens, manifestSum, `round ${round}: manifest sum mismatch`);
      // Pinned candidates must be present (possibly compressed/truncated) —
      // UNLESS the budget was already exhausted by earlier pinned items
      // (the hard budget cap outranks the pinned guarantee, SPEC §4.3).
      const EXHAUSTED_MARGIN = 25; // ≈ header + ellipsis: no meaningful space left
      for (const p of candidates.filter((c) => c.pinned === 1)) {
        const present = result.manifest.some((m) => m.id === p.id);
        if (!present) {
          assert.ok(
            budget - result.totalTokens <= EXHAUSTED_MARGIN,
            `round ${round}: pinned ${p.id} dropped with ${budget - result.totalTokens} tokens still free`,
          );
        }
      }
    }
  });

  it("pinned survives at budget=150 even with long content (compress/truncate)", async () => {
    const longContent =
      "The user has an elaborate multi-cloud deployment pipeline involving Alibaba Cloud ECS, " +
      "GitHub Actions, pm2 process management, Caddy TLS termination, SQLite WAL persistence, " +
      "Redis rate limiting, and a strict rule that production deploys only happen before noon " +
      "on weekdays after all ninety-six tests pass in CI and the demo environment is verified.";
    const candidates: ScoredCandidate[] = [
      makeCandidate("pin1", longContent, "preference", 0, 1, "session-1"),
      makeCandidate("f1", "The user likes green tea.", "fact", 0.9),
    ];

    const result = await packContext(candidates, 150);

    assert.ok(result.manifest.some((m) => m.id === "pin1"), "pinned must survive");
    assert.ok(result.totalTokens <= 150, "hard budget holds");
    assert.equal(result.manifest[0].id, "pin1", "pinned packs first");
    assert.equal(result.manifest[0].utility, Infinity);
  });

  it("reserved sub-budget: preferences/decisions pack before higher-utility facts", async () => {
    const candidates: ScoredCandidate[] = [
      makeCandidate("fact-hi", "The user's CI pipeline runs on GitHub Actions.", "fact", 0.95),
      makeCandidate("pref-lo", "The user prefers concise answers.", "preference", 0.30),
    ];
    const result = await packContext(candidates, 500);
    const ids = result.manifest.map((m) => m.id);
    assert.deepEqual(ids, ["pref-lo", "fact-hi"], "critical types claim the reserve first");
  });

  it("manifest carries session_born provenance", async () => {
    const candidates: ScoredCandidate[] = [
      makeCandidate("a", "The user prefers TypeScript.", "preference", 0.8, 0, "session-1"),
      makeCandidate("b", "The user is a senior engineer.", "fact", 0.6, 0, null),
    ];
    const result = await packContext(candidates, 200);
    const a = result.manifest.find((m) => m.id === "a");
    const b = result.manifest.find((m) => m.id === "b");
    assert.equal(a?.session_born, "session-1");
    assert.equal(b?.session_born, null);
  });

  it("manifest token counts match the packed lines; text = lines joined", async () => {
    const candidates: ScoredCandidate[] = [
      makeCandidate("x", "Prefers vim keybindings.", "preference", 0.7),
      makeCandidate("y", "Uses macOS on Apple Silicon.", "fact", 0.5),
    ];
    const result = await packContext(candidates, 300);

    for (const entry of result.manifest) {
      const cand = candidates.find((c) => c.id === entry.id)!;
      assert.equal(entry.tokens, countTokens(`[${cand.type}] ${cand.content}`));
    }
    const expectedLines = result.manifest.map((entry) => {
      const cand = candidates.find((c) => c.id === entry.id)!;
      return `[${cand.type}] ${cand.content}`;
    });
    assert.equal(result.text, expectedLines.join("\n"));
  });

  it("ordering within phase 2: by utility descending", async () => {
    const candidates: ScoredCandidate[] = [
      makeCandidate("low",  "Low priority note text.",           "note",  0.10),
      makeCandidate("high", "High utility fact about the user.", "fact",  0.90),
      makeCandidate("mid",  "Medium skill level description.",   "skill", 0.50),
    ];
    const result = await packContext(candidates, 500);
    assert.deepEqual(result.manifest.map((m) => m.id), ["high", "mid", "low"]);
  });
});

// ─── PART A2: compression path (qwen-flash mocked) ───────────────────────────

describe("packContext compression (offline, mocked flash)", () => {
  after(() => { globalThis.fetch = realFetch; });

  it("overflow high-utility item gets compressed to fit", async () => {
    let llmCalls = 0;
    globalThis.fetch = (async () => {
      llmCalls++;
      return new Response(
        JSON.stringify({
          id: "c", object: "chat.completion",
          choices: [{ index: 0, message: { role: "assistant", content: "User deploys via pm2 on Alibaba ECS with Caddy TLS; deploys only before noon after CI passes." }, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;

    const longContent =
      "The user maintains a deployment process where the application is bundled, uploaded to an " +
      "Alibaba Cloud ECS instance in the Singapore region, supervised by pm2 with automatic restart " +
      "policies, fronted by Caddy for TLS, and the user insists deployments happen strictly before " +
      "noon local time and only after the full continuous-integration suite has passed twice.";
    const small = makeCandidate("small", "The user likes green tea.", "note", 0.9);
    const big   = makeCandidate("big", longContent, "fact", 0.8);

    // Budget: fits `small` + ~40 tokens — too small for raw `big` (>60 tokens),
    // big enough that the compression target (remaining - header) ≥ 20.
    const smallTokens = countTokens("[note] The user likes green tea.");
    const budget = smallTokens + 40;

    const result = await packContext([small, big], budget);

    assert.ok(llmCalls >= 1, "compression must be attempted");
    const bigEntry = result.manifest.find((m) => m.id === "big");
    assert.ok(bigEntry, "compressed big item must be packed");
    assert.ok(bigEntry.tokens <= 40, "compressed item must fit the remaining budget");
    assert.ok(result.totalTokens <= budget, "hard budget holds");
  });

  it("compression failure degrades gracefully (item skipped, budget intact)", async () => {
    globalThis.fetch = (async () =>
      new Response("boom", { status: 500 })) as typeof fetch;

    const longContent = "long ".repeat(120) + "story about the user.";
    const result = await packContext(
      [
        makeCandidate("ok", "The user likes green tea.", "note", 0.9),
        makeCandidate("big", longContent, "fact", 0.8),
      ],
      60,
    );

    assert.ok(result.manifest.some((m) => m.id === "ok"));
    assert.ok(!result.manifest.some((m) => m.id === "big"), "uncompressible item is skipped");
    assert.ok(result.totalTokens <= 60);
  });
});
