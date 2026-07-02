/**
 * src/mcp/server.ts
 *
 * HTTP-streamable MCP server — exposes Engram's memory substrate as MCP tools.
 * Backed by the same src/memory/* engine as the REST API; zero logic duplication.
 *
 * Tools:
 *   remember       → writeMemories        store one typed memory
 *   recall         → recall()             hybrid retrieve → score → pack
 *   forget         → updateMemoryStatus   soft-delete by id or by content query
 *   list_memories  → listMemories         filtered listing with pagination
 *   act            → loop.act             full agent loop (recall→decide→reflect→write)
 *   memory_stats   → DB counts + runLog + token usage
 *
 * Transport: WebStandardStreamableHTTPServerTransport (stateless — no sessions).
 * Each POST /mcp request spawns a fresh McpServer bound to a fresh transport,
 * but all share the same underlying Db singleton.
 *
 * Connect URL:  POST http://host:PORT/mcp
 * MCP config:
 *   {
 *     "mcpServers": {
 *       "engram": {
 *         "url": "http://localhost:3000/mcp",
 *         "transport": "streamable-http"
 *       }
 *     }
 *   }
 */

import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { getDb } from "../db/client.js";
import { writeMemories, type MemoryCandidate } from "../memory/write.js";
import { recall as recallFn } from "../memory/read.js";
import { listMemories, updateMemoryStatus, getMemory } from "../db/queries.js";
import { getRunLog } from "../agent/scheduler.js";
import { act as loopAct, trackWrite } from "../agent/loop.js";
import { getUsageStats, estimateCostUSD } from "../llm/qwen.js";
import type { Db } from "../db/client.js";

// ─── Shared Zod fragments ─────────────────────────────────────────────────────

const MemoryTypeEnum = z.enum(["fact", "preference", "decision", "event", "skill", "hypothesis", "note"]);
const MemoryStatusEnum = z.enum(["active", "archived", "superseded", "expired"]);

// ─── Helper: wrap any value as an MCP text result ────────────────────────────

function ok(data: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
}

function err(message: string) {
  return { isError: true as const, content: [{ type: "text" as const, text: JSON.stringify({ error: message }) }] };
}

// ─── Tool registration ────────────────────────────────────────────────────────

function registerTools(mcp: McpServer, db: Db): void {
  // ── remember ────────────────────────────────────────────────────────────────
  mcp.registerTool(
    "remember",
    {
      description:
        "Store a typed memory for a user. The memory is deduplicated against existing ones " +
        "and supersession detection runs automatically after write.",
      inputSchema: {
        userId:     z.string().min(1).describe("Stable user identifier"),
        type:       MemoryTypeEnum.describe("Memory category"),
        content:    z.string().min(5).describe("Self-contained memory statement in third person"),
        salience:   z.number().min(0).max(1).default(0.5).describe("Importance 0–1"),
        confidence: z.number().min(0).max(1).default(0.8).describe("Certainty 0–1"),
        tags:       z.array(z.string()).default([]).describe("Topic tags"),
        embedding:  z.array(z.number()).optional().describe("Pre-computed 1024-dim embedding; server computes it if omitted"),
      },
    },
    async (args) => {
      try {
        const candidate: MemoryCandidate = {
          type:       args.type,
          content:    args.content,
          salience:   args.salience,
          confidence: args.confidence,
          tags:       args.tags,
          embedding:  args.embedding,
        };
        const result = await writeMemories(args.userId, null, [candidate], db);
        return ok({
          inserted: result.inserted,
          skipped:  result.skipped,
          summary:  result.inserted.length > 0
            ? `Stored memory ${result.inserted[0]}`
            : `Skipped (${result.skipped[0]?.reason ?? "unknown"})`,
        });
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  );

  // ── recall ──────────────────────────────────────────────────────────────────
  mcp.registerTool(
    "recall",
    {
      description:
        "Retrieve and pack the most decision-relevant memories for a query. " +
        "Returns a packed context string (ready for prompt injection) and a manifest.",
      inputSchema: {
        userId: z.string().min(1).describe("Stable user identifier"),
        query:  z.string().min(1).describe("Natural-language query or current user turn"),
        budget: z
          .number()
          .int()
          .positive()
          .optional()
          .describe("Token budget for packed context (default: ENGRAM_CONTEXT_BUDGET)"),
      },
    },
    async (args) => {
      try {
        const packed = await recallFn(args.userId, args.query, db, args.budget);
        return ok({
          context:     packed.text,
          manifest:    packed.manifest,
          totalTokens: packed.totalTokens,
        });
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  );

  // ── forget ──────────────────────────────────────────────────────────────────
  mcp.registerTool(
    "forget",
    {
      description:
        "Soft-delete a memory by id (status → archived) or find and archive the best " +
        "match for a content query. Never hard-deletes — audit trail is preserved.",
      inputSchema: {
        userId: z.string().min(1).describe("Stable user identifier"),
        id:     z.string().optional().describe("Exact memory ID to archive"),
        query:  z.string().optional().describe("Content query — archives the top-matching active memory"),
        reason: z.string().optional().describe("Optional reason recorded in status change"),
      },
    },
    async (args) => {
      try {
        if (!args.id && !args.query) {
          return err("Provide either id or query");
        }

        let targetId = args.id;

        if (!targetId && args.query) {
          // Find best semantic match and archive it
          const { hybridRetrieve } = await import("../memory/read.js");
          const results = await hybridRetrieve(args.userId, args.query, db, {
            k: 1,
            skipRerank: true,
          });
          if (results.length === 0) return err("No matching memory found");
          targetId = results[0].id;
        }

        const mem = getMemory(db, targetId!);
        if (!mem) return err(`Memory ${targetId} not found`);
        if (mem.user_id !== args.userId) return err("Memory belongs to a different user");

        updateMemoryStatus(db, targetId!, "archived");
        // Confirmation echo: callers see exactly WHAT was forgotten.
        return ok({
          archived:       targetId,
          content:        mem.content,
          type:           mem.type,
          previousStatus: mem.status,
          reason:         args.reason ?? null,
        });
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  );

  // ── list_memories ────────────────────────────────────────────────────────────
  mcp.registerTool(
    "list_memories",
    {
      description: "List stored memories for a user with optional type/status filtering.",
      inputSchema: {
        userId: z.string().min(1).describe("Stable user identifier"),
        type:   MemoryTypeEnum.optional().describe("Filter by memory type"),
        status: MemoryStatusEnum.optional().describe("Filter by status (default: active)"),
        pinned: z.boolean().optional().describe("Filter pinned/unpinned"),
        limit:  z.number().int().positive().max(200).default(20).describe("Max results"),
        offset: z.number().int().min(0).default(0).describe("Pagination offset"),
      },
    },
    async (args) => {
      try {
        const memories = listMemories(db, {
          user_id: args.userId,
          type:    args.type,
          status:  args.status ?? "active",
          pinned:  args.pinned,
          limit:   args.limit,
          offset:  args.offset,
        });

        return ok({
          memories: memories.map((m) => ({
            id:         m.id,
            type:       m.type,
            status:     m.status,
            content:    m.content,
            salience:   m.salience,
            confidence: m.confidence,
            pinned:     m.pinned === 1,
            tags:       JSON.parse(m.tags) as string[],
            created_at: new Date(m.created_at).toISOString(),
            access_count: m.access_count,
          })),
          count: memories.length,
        });
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  );

  // ── act ─────────────────────────────────────────────────────────────────────
  mcp.registerTool(
    "act",
    {
      description:
        "Run the full Engram agent loop for a conversation turn: " +
        "recall relevant memories → decide with qwen-plus → " +
        "return the answer and asynchronously write new memories extracted from the turn.",
      inputSchema: {
        userId:    z.string().min(1).describe("Stable user identifier"),
        sessionId: z.string().min(1).describe("Conversation session ID"),
        turn:      z.string().min(1).describe("The user's current input text"),
        budget:    z.number().int().positive().optional().describe(
          "Token budget for recalled context (default: ENGRAM_CONTEXT_BUDGET)",
        ),
      },
    },
    async (args) => {
      try {
        const result = await loopAct(args.userId, args.sessionId, args.turn, db, {
          budget: args.budget,
        });

        // Register the background write (extract+writeMemories) so it drains cleanly.
        trackWrite(result.writePromise);

        return ok({
          answer:          result.answer,
          contextManifest: result.contextManifest,
          contextTokens:   result.contextTokens,
          usage:           result.usage,
          model:           result.model,
        });
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  );

  // ── memory_stats ─────────────────────────────────────────────────────────────
  mcp.registerTool(
    "memory_stats",
    {
      description: "Return memory counts by type/status and the last scheduler run log.",
      inputSchema: {
        userId: z.string().min(1).describe("Stable user identifier"),
      },
    },
    async (args) => {
      try {
        const byStatus = db
          .prepare(
            `SELECT status, COUNT(*) as count FROM memories WHERE user_id = ? GROUP BY status`,
          )
          .all(args.userId) as { status: string; count: number }[];

        const byType = db
          .prepare(
            `SELECT type, COUNT(*) as count FROM memories WHERE user_id = ? AND status = 'active' GROUP BY type`,
          )
          .all(args.userId) as { type: string; count: number }[];

        const recentRuns = getRunLog()
          .filter((e) => e.userId === args.userId)
          .slice(-5)
          .map((e) => ({
            ts:             new Date(e.ts).toISOString(),
            consolidated:   e.consolidated,
            decayed:        e.decayed,
            inferred:       e.inferred,
            durationMs:     e.durationMs,
            skipped:        e.skipped,
          }));

        const sleepRuns = db
          .prepare("SELECT * FROM sleep_runs WHERE user_id = ? ORDER BY run_at DESC LIMIT 5")
          .all(args.userId) as Record<string, unknown>[];

        const tokenUsage = getUsageStats();

        return ok({
          byStatus,
          byType,
          sleepRuns,
          recentSchedulerRuns: recentRuns,
          tokenUsage,
          estimatedCostUSD: estimateCostUSD(),
        });
      } catch (e) {
        return err(e instanceof Error ? e.message : String(e));
      }
    },
  );
}

// ─── McpServer factory ────────────────────────────────────────────────────────

/**
 * Create a fully-wired McpServer backed by the given Db.
 * Exported for in-process testing with InMemoryTransport.
 */
export function createMcpServer(db: Db): McpServer {
  const mcp = new McpServer(
    { name: "engram-memory", version: "0.1.0" },
    { capabilities: { tools: {} } },
  );
  registerTools(mcp, db);
  return mcp;
}

// ─── HTTP handler (stateless, one transport per request) ─────────────────────

/**
 * Handle a single MCP HTTP request.
 * Compatible with Hono (Web Standard Request → Response).
 *
 * Add to your Hono app:
 *   app.all("/mcp", (c) => handleMcpRequest(c.req.raw));
 */
export async function handleMcpRequest(req: Request): Promise<Response> {
  const db = getDb();
  const mcp = createMcpServer(db);

  const transport = new WebStandardStreamableHTTPServerTransport({
    sessionIdGenerator: undefined, // stateless — no session tracking
  });

  await mcp.connect(transport);
  return transport.handleRequest(req);
}
