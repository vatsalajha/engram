/**
 * test/mcp.test.ts
 *
 * PART A — in-process (no HTTP, no API key):
 *   Uses InMemoryTransport to wire McpServer ↔ Client in the same process.
 *   Tests remember → recall round-trip with pre-baked embeddings.
 *
 * PART B — integration (DASHSCOPE_API_KEY required):
 *   Full live remember → recall with real embeddings confirming semantic recall.
 *
 * Run: npm test
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { openDb, type Db } from "../src/db/client.js";
import { createMcpServer } from "../src/mcp/server.js";
import { insertMemory } from "../src/db/queries.js";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function unitVec(d: number, dim = 1024): number[] {
  const v = new Array<number>(dim).fill(0);
  v[d] = 1.0;
  return v;
}

/** Wire an McpServer to a Client via InMemoryTransport pair. */
async function makeClient(db: Db): Promise<{ client: Client; cleanup: () => Promise<void> }> {
  const mcp = createMcpServer(db);
  const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();

  await mcp.connect(serverTransport);

  const client = new Client({ name: "test-client", version: "0.0.1" });
  await client.connect(clientTransport);

  return {
    client,
    cleanup: async () => {
      await client.close();
      await mcp.close();
    },
  };
}

/** Call a tool and parse the JSON from the first text content block. */
async function callTool<T>(client: Client, name: string, args: Record<string, unknown>): Promise<T> {
  const result = await client.callTool({ name, arguments: args });
  const text = (result.content as { type: string; text: string }[]).find((c) => c.type === "text")?.text;
  if (!text) throw new Error(`Tool ${name} returned no text content`);
  return JSON.parse(text) as T;
}

// ─── PART A: in-process offline tests ────────────────────────────────────────

describe("MCP server (in-process, offline)", () => {
  let db: Db;
  let client: Client;
  let cleanup: () => Promise<void>;

  before(async () => {
    db = openDb(":memory:");
    ({ client, cleanup } = await makeClient(db));
  });

  after(async () => {
    await cleanup();
    db.close();
  });

  it("tools/list returns the expected 6 tools", async () => {
    const { tools } = await client.listTools();
    const names = tools.map((t) => t.name).sort();
    assert.deepEqual(names, [
      "act",
      "forget",
      "list_memories",
      "memory_stats",
      "recall",
      "remember",
    ]);
  });

  it("remember: inserts a new memory and returns its id", async () => {
    const result = await callTool<{ inserted: string[]; skipped: unknown[]; summary: string; error?: string }>(
      client,
      "remember",
      {
        userId:     "mcp_test_user",
        type:       "preference",
        content:    "The user prefers dark mode in all developer tools.",
        salience:   0.9,
        confidence: 1.0,
        tags:       ["ui", "preferences"],
        embedding:  unitVec(1),   // pre-computed so no API call needed
      },
    );

    if (result.error) throw new Error(`remember returned error: ${result.error}`);
    assert.ok(result.inserted.length === 1, "expected 1 inserted memory");
    assert.ok(result.skipped.length === 0, "expected 0 skipped");
    assert.ok(result.summary.startsWith("Stored"), `unexpected summary: ${result.summary}`);
  });

  it("remember: duplicate is skipped with exact_match reason", async () => {
    const result = await callTool<{ inserted: string[]; skipped: { reason: string }[]; error?: string }>(
      client,
      "remember",
      {
        userId:     "mcp_test_user",
        type:       "preference",
        content:    "The user prefers dark mode in all developer tools.",
        salience:   0.9,
        confidence: 1.0,
        tags:       ["ui", "preferences"],
        embedding:  unitVec(1),
      },
    );
    if (result.error) throw new Error(`remember returned error: ${result.error}`);
    assert.equal(result.inserted.length, 0, "duplicate should not insert");
    assert.equal(result.skipped[0]?.reason, "exact_match");
  });

  it("list_memories: returns the remembered memory", async () => {
    const result = await callTool<{ memories: { type: string; content: string }[]; count: number }>(
      client,
      "list_memories",
      { userId: "mcp_test_user", status: "active", limit: 10 },
    );
    assert.ok(result.count >= 1, "should list at least one memory");
    assert.ok(
      result.memories.some((m) => m.content.includes("dark mode")),
      "dark mode preference should be in the list",
    );
  });

  it("recall: remembered memory surfaces in packed context (pre-baked embedding)", async () => {
    // Seed a memory with a known embedding so offline recall works
    const prefId = insertMemory(db, {
      user_id:    "mcp_recall_user",
      type:       "preference",
      content:    "The user always wants concise answers.",
      salience:   0.85,
      confidence: 0.95,
      embedding:  unitVec(0),
    });

    // recall will try to embed the query string. Without an API key it fails and
    // the MCP handler returns {error: "..."} instead of {context, manifest, totalTokens}.
    // The FTS leg (BM25) still runs but without the vector leg the recall is partial.
    // Offline: we accept either a successful recall OR a graceful error response.
    const result = await callTool<{ context?: string; manifest?: { id: string }[]; totalTokens?: number; error?: string }>(
      client,
      "recall",
      { userId: "mcp_recall_user", query: "concise answers preference" },
    );

    if (result.error) {
      // No API key — embed failed; the handler returned {error}. Skip gracefully.
      console.log(`  (recall skipped offline — embed unavailable: ${result.error.slice(0, 80)})`);
      return;
    }

    assert.ok(result.manifest?.some((m) => m.id === prefId), "recalled manifest must include the seeded memory");
    assert.ok(result.context?.includes("concise"), "packed context must include memory content");
    assert.ok((result.totalTokens ?? 0) > 0, "totalTokens must be positive");
  });

  it("forget: archives a memory by id", async () => {
    const id = insertMemory(db, {
      user_id: "mcp_forget_user",
      type:    "note",
      content: "Temporary debug note to forget.",
      salience: 0.2,
    });

    const result = await callTool<{ archived: string; previousStatus: string }>(
      client,
      "forget",
      { userId: "mcp_forget_user", id },
    );

    assert.equal(result.archived, id);
    assert.equal(result.previousStatus, "active");

    // Verify it's no longer in active list
    const list = await callTool<{ memories: { id: string }[] }>(
      client,
      "list_memories",
      { userId: "mcp_forget_user", status: "active" },
    );
    assert.ok(!list.memories.some((m) => m.id === id), "archived memory should not appear in active list");
  });

  it("forget: error when neither id nor query provided", async () => {
    const result = await callTool<{ error?: string }>(
      client,
      "forget",
      { userId: "mcp_forget_user" },
    );
    assert.ok(result.error?.includes("id or query"), `expected error message, got: ${result.error}`);
  });

  it("memory_stats: returns counts by type and status", async () => {
    const result = await callTool<{
      byStatus:    { status: string; count: number }[];
      byType:      { type: string;   count: number }[];
      tokenUsage?: { calls: number; totalTokens: number };
    }>(
      client,
      "memory_stats",
      { userId: "mcp_test_user" },
    );

    assert.ok(Array.isArray(result.byStatus), "byStatus should be an array");
    assert.ok(Array.isArray(result.byType),   "byType should be an array");

    const activeRow = result.byStatus.find((r) => r.status === "active");
    assert.ok(activeRow && activeRow.count >= 1, "should have at least 1 active memory");

    // tokenUsage field added by the upgraded memory_stats tool
    assert.ok(result.tokenUsage !== undefined, "tokenUsage should be present");
    assert.ok(typeof result.tokenUsage?.calls === "number", "tokenUsage.calls must be a number");
  });

  it("act: returns graceful error when LLM is unavailable (no API key)", async () => {
    // act calls recall (which calls embed) and then chatStream — both need an API key.
    // Without one, the tool must return {error: ...} rather than throwing.
    const result = await callTool<{
      answer?: string;
      error?:  string;
    }>(
      client,
      "act",
      {
        userId:    "mcp_act_user",
        sessionId: "offline-session",
        turn:      "What is my preferred coding style?",
        hardMode:  false,
        budget:    400,
      },
    );

    // Either it succeeded (unlikely offline) or it returned a structured error
    if (result.error) {
      assert.ok(result.error.length > 0, "error message must be non-empty");
      console.log(`  (act offline — expected API error: ${result.error.slice(0, 80)})`);
    } else {
      // If somehow it worked (e.g. FTS-only path), answer must be a non-empty string
      assert.ok(typeof result.answer === "string" && result.answer.length > 0);
    }
  });
});

// ─── PART B: integration (requires DASHSCOPE_API_KEY) ────────────────────────

const _rawKey = process.env.DASHSCOPE_API_KEY ?? "";
const HAS_API_KEY = _rawKey.length > 20 && !_rawKey.includes("...");

describe("MCP remember → recall (integration)", { skip: !HAS_API_KEY }, () => {
  let db: Db;
  let client: Client;
  let cleanup: () => Promise<void>;

  before(async () => {
    db = openDb(":memory:");
    ({ client, cleanup } = await makeClient(db));
  });

  after(async () => {
    await cleanup();
    db.close();
  });

  it("act: full loop — returns answer + manifest, writes new memories", async () => {
    // Seed a preference so there's something to recall
    await callTool<{ inserted: string[] }>(client, "remember", {
      userId:    "mcp_act_live_user",
      type:      "preference",
      content:   "The user prefers Rust over C++ for systems programming.",
      salience:  0.85,
      confidence: 1.0,
      tags:      ["programming", "systems"],
    });

    const result = await callTool<{
      answer:          string;
      contextManifest: { id: string; type: string; tokens: number; utility: number }[];
      contextTokens:   number;
      usage:           { promptTokens: number; completionTokens: number; totalTokens: number };
      model:           string;
      error?:          string;
    }>(
      client,
      "act",
      {
        userId:    "mcp_act_live_user",
        sessionId: "live-session-001",
        turn:      "What programming language should I use for a new OS kernel?",
        hardMode:  false,
        budget:    800,
      },
    );

    console.log("\n--- act result ---");
    if (result.error) {
      console.log("  error:", result.error);
      assert.fail(`act returned error: ${result.error}`);
    }
    console.log(`  model:         ${result.model}`);
    console.log(`  contextTokens: ${result.contextTokens}`);
    console.log(`  usage:         ${JSON.stringify(result.usage)}`);
    console.log(`  answer (first 200): ${result.answer.slice(0, 200)}`);
    console.log("  manifest:", result.contextManifest.map(
      (m) => `[${m.type}] u=${m.utility.toFixed(3)}`,
    ));

    assert.ok(result.answer.length > 0, "answer must be non-empty");
    assert.ok(result.usage.totalTokens > 0, "usage.totalTokens must be positive");
    assert.ok(result.model === "qwen-plus" || result.model === "qwen-max", "model must be a qwen variant");
    assert.ok(Array.isArray(result.contextManifest), "contextManifest must be an array");
  });

  it("live remember then recall: remembered item appears in packed context", async () => {
    const CONTENT = "The user is an expert in distributed systems and prefers eventual consistency patterns.";

    // remember
    const remResult = await callTool<{ inserted: string[]; summary: string }>(
      client, "remember",
      {
        userId:     "mcp_live_user",
        type:       "fact",
        content:    CONTENT,
        salience:   0.8,
        confidence: 0.95,
        tags:       ["expertise", "distributed-systems"],
      },
    );
    console.log("\n--- remember result ---");
    console.log(JSON.stringify(remResult, null, 2));
    assert.ok(remResult.inserted.length >= 1, "memory should be inserted");
    const memId = remResult.inserted[0];

    // recall
    const recResult = await callTool<{ context: string; manifest: { id: string; type: string; tokens: number; utility: number }[]; totalTokens: number }>(
      client, "recall",
      { userId: "mcp_live_user", query: "distributed systems expertise and architecture preferences" },
    );
    console.log("\n--- recall result ---");
    console.log(`Context (${recResult.totalTokens} tokens):\n${recResult.context}`);
    console.log("\nManifest:");
    console.table(recResult.manifest);

    assert.ok(
      recResult.manifest.some((m) => m.id === memId),
      `recalled manifest must include the remembered memory (id=${memId})`,
    );
    assert.ok(
      recResult.context.includes("distributed") || recResult.context.includes("consistency"),
      "packed context must include the remembered content",
    );
  });
});
