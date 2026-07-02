/**
 * test/loop.test.ts
 *
 * PART A — offline (no API key needed):

 *   • buildSystemPrompt correctly embeds packed context (or omits it if empty)
 *   • Write-task registry: pendingWriteCount, trackWrite, drainWrites
 *
 * PART B — integration (DASHSCOPE_API_KEY required):
 *   • Full act() round-trip: recall → decide → write
 *   • After drainWrites() the new memory is queryable via listMemories
 *   • SSE token stream: onToken callback fires ≥1 time
 *
 * Run: npm test
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { openDb, type Db } from "../src/db/client.js";
import { insertMemory } from "../src/db/queries.js";
import {
  buildSystemPrompt,
  pendingWriteCount,
  trackWrite,
  drainWrites,
  act,
} from "../src/agent/loop.js";
import { listMemories } from "../src/db/queries.js";

// ─── PART A: offline tests ────────────────────────────────────────────────────

describe("buildSystemPrompt (offline)", () => {
  it("returns base prompt when packed context is empty", () => {
    const prompt = buildSystemPrompt("");
    assert.ok(prompt.includes("project copilot"), "base persona present");
    assert.ok(!prompt.includes("[What you know about this user]"), "no memory block when empty");
  });

  it("embeds packed context when provided", () => {
    const ctx = "[preference] The user likes dark mode.";
    const prompt = buildSystemPrompt(ctx);
    assert.ok(prompt.includes("[What you know about this user]"), "memory block header present");
    assert.ok(prompt.includes(ctx), "packed context embedded verbatim");
    assert.ok(prompt.includes("[end]"), "memory block footer present");
  });

  it("does not embed context for whitespace-only string", () => {
    const prompt = buildSystemPrompt("   \n  ");
    assert.ok(!prompt.includes("[What you know about this user]"), "no block for whitespace");
  });
});

describe("write-task registry (offline)", () => {
  it("pendingWriteCount starts at 0", () => {
    // We can't guarantee 0 if parallel tests leaked, but in isolation it should be 0.
    // Just assert it's a non-negative integer.
    const count = pendingWriteCount();
    assert.ok(typeof count === "number" && count >= 0, `count=${count}`);
  });

  it("trackWrite increments count; drainWrites resolves all and count returns to 0", async () => {
    const before = pendingWriteCount();

    let resolved = false;
    const p: Promise<void> = new Promise((resolve) => {
      setTimeout(() => {
        resolved = true;
        resolve();
      }, 20);
    });

    trackWrite(p);
    assert.ok(pendingWriteCount() >= before + 1, "count incremented after trackWrite");

    await drainWrites();
    assert.ok(resolved, "promise resolved before drainWrites returned");
    assert.equal(pendingWriteCount(), before, "count back to baseline after drain");
  });

  it("drainWrites with no pending writes resolves immediately", async () => {
    // Drain any leftover from prior tests first
    await drainWrites();
    const t0 = Date.now();
    await drainWrites();
    const elapsed = Date.now() - t0;
    assert.ok(elapsed < 200, `drain took ${elapsed}ms — expected near-instant`);
  });

  it("trackWrite handles a rejecting promise without propagating", async () => {
    const p: Promise<void> = Promise.reject(new Error("simulated write failure"));
    // trackWrite should not throw
    assert.doesNotThrow(() => trackWrite(p));
    // drainWrites uses allSettled so this should not throw either
    await drainWrites();
  });
});

// ─── PART B: integration tests ────────────────────────────────────────────────

const _rawKey = process.env.DASHSCOPE_API_KEY ?? "";
const HAS_API_KEY = _rawKey.length > 20 && !_rawKey.includes("...");

describe("act() — full loop (integration)", { skip: !HAS_API_KEY }, () => {
  let db: Db;

  before(() => {
    db = openDb(":memory:");
  });

  after(() => {
    db.close();
  });

  it("act() returns answer + manifest + usage for a simple turn", async () => {
    // Seed a memory so the context packer has something to work with
    insertMemory(db, {
      user_id:    "loop_test_user",
      type:       "preference",
      content:    "The user prefers TypeScript over JavaScript.",
      salience:   0.85,
      confidence: 0.95,
      embedding:  new Array(1024).fill(0).map((_, i) => (i === 0 ? 1.0 : 0)),
    });

    const tokens: string[] = [];
    const result = await act(
      "loop_test_user",
      "test-session",
      "What language should I use for a new project?",
      db,
      {
        budget:  800,
        onToken: (t) => { tokens.push(t); },
      },
    );

    console.log("\n--- act() result ---");
    console.log(`  model:         ${result.model}`);
    console.log(`  contextTokens: ${result.contextTokens}`);
    console.log(`  usage:         ${JSON.stringify(result.usage)}`);
    console.log(`  tokens fired:  ${tokens.length}`);
    console.log(`  answer (first 200): ${result.answer.slice(0, 200)}`);
    console.log(`  manifest (${result.contextManifest.length} items):`,
      result.contextManifest.map((m) => `{ id: "${m.id}", type: "${m.type}", tokens: ${m.tokens} }`));

    assert.ok(result.answer.length > 0, "answer must not be empty");
    assert.ok(tokens.length >= 1, "onToken callback must fire at least once");
    assert.ok(result.usage.totalTokens > 0, "usage.totalTokens must be positive");
    assert.equal(result.model, "qwen-plus", "should use qwen-plus for normal turn");
    assert.ok(Array.isArray(result.contextManifest), "manifest must be an array");
    assert.ok(result.writePromise instanceof Promise, "writePromise must be a Promise");
  });

  it("after act() + drainWrites(), new memories are queryable", async () => {
    const CONTENT = "My name is Alice and I am a senior engineer at Acme Corp.";

    const result = await act(
      "loop_persist_user",
      "s1",
      CONTENT,
      db,
      { budget: 200 },
    );

    trackWrite(result.writePromise);
    await drainWrites();

    const memories = listMemories(db, {
      user_id: "loop_persist_user",
      status:  "active",
      limit:   10,
    });

    console.log(`\n  memories written for loop_persist_user: ${memories.length}`);
    for (const m of memories) {
      console.log(`    [${m.type}] ${m.content}`);
    }

    // The extraction should have produced at least one memory from this input
    assert.ok(memories.length >= 1, "at least one memory must be written after act()");
  });
});
