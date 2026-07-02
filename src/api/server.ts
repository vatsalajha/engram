/**
 * src/api/server.ts
 *
 * Hono HTTP server — REST API + MCP bridge.
 *
 * Routes
 * ──────
 *   GET  /health            liveness probe (ECS curl proof)
 *   POST /act               agent loop → SSE token stream + manifest + usage
 *   POST /ingest            extract + write only (no answer)
 *   POST /recall            packed context for a query (debug/demo)
 *   GET  /admin/stats       counts + sleep-run log + cumulative Qwen usage/cost
 *   ALL  /mcp               HTTP-streamable MCP server
 *
 * Cross-cutting middleware (POST /act + POST /ingest)
 * ───────────────────────────────────────────────────
 *   • Zod body validation (per-route)
 *   • Per-user sliding-window rate limit  (Redis counter, degrades if Redis down)
 *   • Per-user idempotency key            (Redis SET NX, degrades if Redis down)
 */

import { Hono, type Context } from "hono";
import { cors } from "hono/cors";
import { z } from "zod";
import { streamSSE } from "hono/streaming";
import { Redis } from "ioredis";
import { initDb } from "../db/client.js";
import { config } from "../config.js";
import { extractMemories, writeMemories } from "../memory/write.js";
import { recall } from "../memory/read.js";
import { act, trackWrite, pendingWriteCount, drainWrites } from "../agent/loop.js";
import { applyFeedback } from "../memory/belief.js";
import { runSleep } from "../memory/sleep.js";
import { handleMcpRequest } from "../mcp/server.js";
import { getRunLog, startScheduler } from "../agent/scheduler.js";
import { getUsageStats, estimateCostUSD } from "../llm/qwen.js";

// ─── Infrastructure singletons ────────────────────────────────────────────────

const db = initDb(config.ENGRAM_DB_PATH);
startScheduler(db);

// Redis client — shared between rate-limit and idempotency middleware.
// Optional: only connected when REDIS_URL is set; the checkRateLimit /
// checkIdempotency helpers fall back to in-process state otherwise.
// lazyConnect so startup doesn't fail when Redis is unavailable.
const redis: Redis | null = config.REDIS_URL
  ? new Redis(config.REDIS_URL, {
      lazyConnect: true,
      enableOfflineQueue: false,
      maxRetriesPerRequest: 0,
    })
  : null;
redis?.on("error", (err: Error) =>
  console.warn("[api] Redis error (rate-limit/idempotency degraded):", err.message),
);

// ─── Rate limit (Redis sliding-window counter, in-process fallback) ──────────

const _rlLocal = new Map<string, number>(); // "user:window" → count

async function checkRateLimit(userId: string): Promise<"ok" | "limited"> {
  const windowMs = config.ENGRAM_RATE_LIMIT_WINDOW_SECONDS * 1000;
  const window = Math.floor(Date.now() / windowMs);
  const key = `engram:rl:${userId}:${window}`;

  if (redis) {
    try {
      const count = await redis.incr(key);
      if (count === 1) {
        // Set expiry to 2× the window so the key is cleaned up automatically
        await redis.expire(key, config.ENGRAM_RATE_LIMIT_WINDOW_SECONDS * 2);
      }
      return count <= config.ENGRAM_RATE_LIMIT_MAX ? "ok" : "limited";
    } catch {
      return "ok"; // degrade gracefully when Redis is unavailable
    }
  }

  // In-process fallback (single-instance default per SPEC §2)
  const count = (_rlLocal.get(key) ?? 0) + 1;
  _rlLocal.set(key, count);
  if (_rlLocal.size > 10_000) {
    // Drop counters from past windows so the map can't grow unbounded
    const prefixNow = `:${window}`;
    for (const k of _rlLocal.keys()) if (!k.endsWith(prefixNow)) _rlLocal.delete(k);
  }
  return count <= config.ENGRAM_RATE_LIMIT_MAX ? "ok" : "limited";
}

// ─── Idempotency (Redis SET NX, in-process fallback) ─────────────────────────
// JSON routes cache their response body so an idempotent replay returns the
// ORIGINAL result (200 + Idempotency-Replayed header) instead of an error.
// /act streams SSE, so a duplicate there is rejected with 409 instead.

interface IdemRecord {
  expiry: number;
  response?: unknown; // cached JSON body, set after the handler succeeds
}

const _idemLocal = new Map<string, IdemRecord>();

type IdemState = { state: "new" } | { state: "duplicate"; cached: unknown | null };

async function checkIdempotency(userId: string, idemKey: string): Promise<IdemState> {
  const key = `engram:idem:${userId}:${idemKey}`;

  if (redis) {
    try {
      const result = await redis.set(key, "pending", "EX", config.ENGRAM_IDEMPOTENCY_TTL_SECONDS, "NX");
      if (result === "OK") return { state: "new" };
      const raw = await redis.get(key);
      return {
        state: "duplicate",
        cached: raw && raw !== "pending" ? (JSON.parse(raw) as unknown) : null,
      };
    } catch {
      return { state: "new" }; // degrade gracefully when Redis is unavailable
    }
  }

  // In-process fallback
  const now = Date.now();
  const existing = _idemLocal.get(key);
  if (existing && existing.expiry > now) {
    return { state: "duplicate", cached: existing.response ?? null };
  }
  _idemLocal.set(key, { expiry: now + config.ENGRAM_IDEMPOTENCY_TTL_SECONDS * 1000 });
  if (_idemLocal.size > 50_000) {
    for (const [k, rec] of _idemLocal) if (rec.expiry <= now) _idemLocal.delete(k);
  }
  return { state: "new" };
}

/** Attach the handler's JSON result to an idempotency key for future replays. */
async function storeIdempotentResult(userId: string, idemKey: string, response: unknown): Promise<void> {
  const key = `engram:idem:${userId}:${idemKey}`;
  if (redis) {
    try {
      await redis.set(key, JSON.stringify(response), "EX", config.ENGRAM_IDEMPOTENCY_TTL_SECONDS);
    } catch { /* best-effort */ }
    return;
  }
  const rec = _idemLocal.get(key);
  if (rec) rec.response = response;
}

// ─── App ──────────────────────────────────────────────────────────────────────

const app = new Hono();

// CORS — allow the Vite dev server and the bundled web/ app to hit the API
app.use(
  "*",
  cors({
    origin: ["http://localhost:5173", "http://localhost:3000"],
    allowHeaders: ["Content-Type", "Idempotency-Key"],
    allowMethods: ["GET", "POST", "OPTIONS"],
  }),
);

// ── GET /health ───────────────────────────────────────────────────────────────

app.get("/health", (c) =>
  c.json({
    ok:      true,
    service: "engram",
    ts:      new Date().toISOString(),
    pending: pendingWriteCount(),
  }),
);

// ── POST /act ─────────────────────────────────────────────────────────────────

const ActBody = z.object({
  userId:    z.string().min(1),
  sessionId: z.string().min(1),
  input:     z.string().min(1),
  budget:    z.number().int().positive().optional(),
});

app.post("/act", async (c) => {
  // ── Validate body ─────────────────────────────────────────────────────────
  const body = await c.req.json().catch(() => null);
  const parsed = ActBody.safeParse(body);
  if (!parsed.success) {
    return c.json({ error: parsed.error.flatten() }, 400);
  }
  const { userId, sessionId, input, budget } = parsed.data;

  // ── Rate limit ────────────────────────────────────────────────────────────
  const rlStatus = await checkRateLimit(userId);
  if (rlStatus === "limited") {
    return c.json(
      { error: "rate_limit_exceeded", retryAfter: config.ENGRAM_RATE_LIMIT_WINDOW_SECONDS },
      429,
    );
  }

  // ── Idempotency (SSE cannot replay a cached stream → 409 on duplicate) ─────
  const idemKey = c.req.header("Idempotency-Key");
  if (idemKey) {
    const idem = await checkIdempotency(userId, idemKey);
    if (idem.state === "duplicate") {
      return c.json({ error: "duplicate_request", idempotencyKey: idemKey }, 409);
    }
  }

  // ── Stream SSE ────────────────────────────────────────────────────────────
  return streamSSE(c, async (stream) => {
    try {
      const result = await act(userId, sessionId, input, db, {
        onToken: (token) => stream.writeSSE({ event: "token", data: token }),
        budget,
      });

      // Track post-response work (guaranteed to complete)
      trackWrite(result.writePromise);

      // Send structured events after all tokens
      await stream.writeSSE({
        event: "manifest",
        data:  JSON.stringify(result.contextManifest),
      });
      await stream.writeSSE({
        event: "usage",
        data:  JSON.stringify({
          ...result.usage,
          model:         result.model,
          contextTokens: result.contextTokens,
        }),
      });
      // Final event carries the whole envelope (SPEC §2): turnId for /feedback,
      // manifest for provenance, usage for the cost counter.
      await stream.writeSSE({
        event: "done",
        data:  JSON.stringify({
          turnId:   result.turnId,
          manifest: result.contextManifest,
          usage:    { ...result.usage, model: result.model, contextTokens: result.contextTokens },
        }),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      await stream.writeSSE({ event: "error", data: JSON.stringify({ error: msg }) });
    }
  });
});

// ── POST /ingest ──────────────────────────────────────────────────────────────

const IngestBody = z.object({
  userId:    z.string().min(1),
  turn:      z.string().min(1),
  assistant: z.string().optional(),
  sessionId: z.string().optional(),
});

app.post("/ingest", async (c) => {
  const body = await c.req.json().catch(() => null);
  const parsed = IngestBody.safeParse(body);
  if (!parsed.success) return c.json({ error: parsed.error.flatten() }, 400);

  const { userId, turn, assistant, sessionId } = parsed.data;

  const rlStatus = await checkRateLimit(userId);
  if (rlStatus === "limited") {
    return c.json(
      { error: "rate_limit_exceeded", retryAfter: config.ENGRAM_RATE_LIMIT_WINDOW_SECONDS },
      429,
    );
  }

  const idemKey = c.req.header("Idempotency-Key");
  if (idemKey) {
    const idem = await checkIdempotency(userId, idemKey);
    if (idem.state === "duplicate") {
      if (idem.cached !== null) {
        c.header("Idempotency-Replayed", "true");
        return c.json(idem.cached as Record<string, unknown>);
      }
      return c.json({ error: "duplicate_request", idempotencyKey: idemKey }, 409);
    }
  }

  const candidates = await extractMemories(turn, assistant);
  const result     = await writeMemories(userId, sessionId ?? null, candidates, db);
  if (idemKey) await storeIdempotentResult(userId, idemKey, result);
  return c.json(result);
});

// ── POST /recall ──────────────────────────────────────────────────────────────

const RecallBody = z.object({
  userId: z.string().min(1),
  query:  z.string().min(1),
  budget: z.number().int().positive().optional(),
});

app.post("/recall", async (c) => {
  const body = await c.req.json().catch(() => null);
  const parsed = RecallBody.safeParse(body);
  if (!parsed.success) return c.json({ error: parsed.error.flatten() }, 400);

  const { userId, query, budget } = parsed.data;
  const packed = await recall(userId, query, db, budget);
  return c.json(packed);
});

// ── POST /feedback ────────────────────────────────────────────────────────────
// Explicit belief signal: the user (or eval harness) confirms/contradicts the
// memories a turn relied on. memoryIds defaults come from the /act done-event
// manifest on the client side.

const FeedbackBody = z.object({
  userId:    z.string().min(1),
  turnId:    z.string().min(1).optional(),
  outcome:   z.enum(["confirmed", "contradicted"]),
  memoryIds: z.array(z.string().min(1)).min(1),
});

app.post("/feedback", async (c) => {
  const body = await c.req.json().catch(() => null);
  const parsed = FeedbackBody.safeParse(body);
  if (!parsed.success) return c.json({ error: parsed.error.flatten() }, 400);

  const { userId, turnId, outcome, memoryIds } = parsed.data;

  const rlStatus = await checkRateLimit(userId);
  if (rlStatus === "limited") {
    return c.json(
      { error: "rate_limit_exceeded", retryAfter: config.ENGRAM_RATE_LIMIT_WINDOW_SECONDS },
      429,
    );
  }

  const idemKey = c.req.header("Idempotency-Key");
  if (idemKey) {
    const idem = await checkIdempotency(userId, idemKey);
    if (idem.state === "duplicate") {
      if (idem.cached !== null) {
        c.header("Idempotency-Replayed", "true");
        return c.json(idem.cached as Record<string, unknown>);
      }
      return c.json({ error: "duplicate_request", idempotencyKey: idemKey }, 409);
    }
  }

  const results = applyFeedback(db, userId, turnId ?? null, outcome, memoryIds);
  const response = { applied: results };
  if (idemKey) await storeIdempotentResult(userId, idemKey, response);
  return c.json(response);
});

// ── POST /sleep ───────────────────────────────────────────────────────────────
// Force one sleep cycle for a user (the demo's "watch it dream" button).

const SleepBody = z.object({ userId: z.string().min(1) });

app.post("/sleep", async (c) => {
  const body = await c.req.json().catch(() => null);
  const parsed = SleepBody.safeParse(body);
  if (!parsed.success) return c.json({ error: parsed.error.flatten() }, 400);

  const result = await runSleep(parsed.data.userId, db);
  return c.json(result, result.skipped ? 409 : 200);
});

// ── GET /admin/stats ──────────────────────────────────────────────────────────

app.get("/admin/stats", (c) => {
  const userId = c.req.query("userId");

  const byStatus = db
    .prepare(
      userId
        ? "SELECT status, COUNT(*) AS count FROM memories WHERE user_id = ? GROUP BY status"
        : "SELECT status, COUNT(*) AS count FROM memories GROUP BY status",
    )
    .all(...(userId ? [userId] : [])) as { status: string; count: number }[];

  const byType = db
    .prepare(
      userId
        ? "SELECT type, COUNT(*) AS count FROM memories WHERE user_id = ? AND status = 'active' GROUP BY type"
        : "SELECT type, COUNT(*) AS count FROM memories WHERE status = 'active' GROUP BY type",
    )
    .all(...(userId ? [userId] : [])) as { type: string; count: number }[];

  const recentRuns = getRunLog()
    .filter((e) => !userId || e.userId === userId)
    .slice(-10)
    .map((e) => ({
      ts:           new Date(e.ts).toISOString(),
      userId:       e.userId,
      consolidated: e.consolidated,
      decayed:      e.decayed,
      inferred:     e.inferred,
      durationMs:   e.durationMs,
      skipped:      e.skipped,
    }));

  const sleepRuns = db
    .prepare(
      userId
        ? "SELECT * FROM sleep_runs WHERE user_id = ? ORDER BY run_at DESC LIMIT 20"
        : "SELECT * FROM sleep_runs ORDER BY run_at DESC LIMIT 20",
    )
    .all(...(userId ? [userId] : [])) as Record<string, unknown>[];

  const tokenUsage    = getUsageStats();
  const estimatedCost = estimateCostUSD();
  const pendingWrites = pendingWriteCount();

  return c.json({
    byStatus,
    byType,
    sleepRuns,
    recentSchedulerRuns: recentRuns,
    tokenUsage,
    estimatedCostUSD: estimatedCost,
    pendingWrites,
  });
});

// ── GET /memories (alias: /api/memories) ─────────────────────────────────────
// Used by the web demo's Memory side panel and the provenance graph UI.
// ?userId= | ?user_id=   required
// ?status=   active|archived|superseded|expired  (default: active + superseded)
// ?limit=    max rows (default 50)
// Response includes every edge whose src AND dst are in the returned set.

function memoriesHandler(c: Context) {
  const userId = c.req.query("userId") ?? c.req.query("user_id");
  if (!userId) return c.json({ error: "userId is required" }, 400);

  const status = c.req.query("status");
  const limit  = Math.min(parseInt(c.req.query("limit") ?? "50", 10) || 50, 200);

  const rows = db
    .prepare(
      status
        ? `SELECT id, type, status, content, salience, confidence, pinned, tags,
                  created_at, access_count, session_id, needs_review, superseded_by
             FROM memories
            WHERE user_id = ? AND status = ?
            ORDER BY created_at DESC
            LIMIT ?`
        : `SELECT id, type, status, content, salience, confidence, pinned, tags,
                  created_at, access_count, session_id, needs_review, superseded_by
             FROM memories
            WHERE user_id = ? AND status IN ('active','superseded')
            ORDER BY created_at DESC
            LIMIT ?`,
    )
    .all(...(status ? [userId, status, limit] : [userId, limit])) as {
      id: string; type: string; status: string; content: string;
      salience: number; confidence: number; pinned: number; tags: string;
      created_at: number; access_count: number; session_id: string | null;
      needs_review: number; superseded_by: string | null;
    }[];

  // Edges among the returned memories (both endpoints visible → drawable).
  const ids = rows.map((r) => r.id);
  let edges: { src_id: string; dst_id: string; kind: string }[] = [];
  if (ids.length > 0) {
    const ph = ids.map(() => "?").join(",");
    edges = db
      .prepare(
        `SELECT src_id, dst_id, kind FROM memory_edges
          WHERE src_id IN (${ph}) AND dst_id IN (${ph})`,
      )
      .all(...ids, ...ids) as { src_id: string; dst_id: string; kind: string }[];
  }

  return c.json({
    memories: rows.map((m) => ({
      ...m,
      pinned:       m.pinned === 1,
      needs_review: m.needs_review === 1,
      tags:         JSON.parse(m.tags ?? "[]") as string[],
      created_at:   new Date(m.created_at).toISOString(),
    })),
    edges,
    count: rows.length,
  });
}

app.get("/memories", memoriesHandler);
app.get("/api/memories", memoriesHandler);

// ── GET /api/eval-runs ────────────────────────────────────────────────────────
// Used by the Dashboard view to render benchmark charts.
// ?arm=no_memory|full_history|engram  (optional, returns all if omitted)
// ?limit=  max rows per arm (default 100)

app.get("/api/eval-runs", (c) => {
  const arm   = c.req.query("arm");
  const limit = Math.min(parseInt(c.req.query("limit") ?? "100", 10) || 100, 500);

  const rows = db
    .prepare(
      arm
        ? `SELECT id, run_at, arm, session_index, task_accuracy, tokens_in_context,
                  recall_at_k, forgetting_precision, latency_p50_ms, latency_p95_ms,
                  cost_usd, metadata
             FROM eval_runs WHERE arm = ? ORDER BY session_index ASC LIMIT ?`
        : `SELECT id, run_at, arm, session_index, task_accuracy, tokens_in_context,
                  recall_at_k, forgetting_precision, latency_p50_ms, latency_p95_ms,
                  cost_usd, metadata
             FROM eval_runs ORDER BY arm ASC, session_index ASC LIMIT ?`,
    )
    .all(...(arm ? [arm, limit] : [limit])) as {
      id: string; run_at: number; arm: string; session_index: number;
      task_accuracy: number | null; tokens_in_context: number | null;
      recall_at_k: number | null; forgetting_precision: number | null;
      latency_p50_ms: number | null; latency_p95_ms: number | null;
      cost_usd: number | null; metadata: string | null;
    }[];

  return c.json({
    runs: rows.map((r) => ({
      ...r,
      run_at:   new Date(r.run_at).toISOString(),
      metadata: r.metadata ? (JSON.parse(r.metadata) as unknown) : null,
    })),
    count: rows.length,
  });
});

// ── MCP bridge ────────────────────────────────────────────────────────────────

app.all("/mcp", (c) => handleMcpRequest(c.req.raw));

// ─── Graceful shutdown ────────────────────────────────────────────────────────

async function shutdown(signal: string): Promise<void> {
  console.log(`[api] ${signal} received — draining pending writes…`);
  await drainWrites();
  redis?.disconnect();
  console.log("[api] shutdown complete");
  process.exit(0);
}

process.once("SIGTERM", () => { void shutdown("SIGTERM"); });
process.once("SIGINT",  () => { void shutdown("SIGINT"); });

// ─── Export ───────────────────────────────────────────────────────────────────

export default app;

// ─── Serve (when run directly) ────────────────────────────────────────────────

const isMain =
  process.argv[1]?.endsWith("server.ts") ||
  process.argv[1]?.endsWith("server.js");

if (isMain) {
  const { serve } = await import("@hono/node-server");
  serve({ fetch: app.fetch, port: config.PORT }, (info) => {
    console.log(`[api] Engram listening on http://localhost:${info.port}`);
  });
}
