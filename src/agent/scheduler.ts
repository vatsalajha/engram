/**
 * src/agent/scheduler.ts
 *
 * Sleep-cycle scheduler — runs runSleep() for every active user on a
 * configurable interval.  Each run is guarded by a per-user SET NX lock:
 * Redis when REDIS_URL is set (multi-instance safe), in-process otherwise.
 *
 * Enable:   ENGRAM_SCHEDULER=on
 * Interval: ENGRAM_SLEEP_INTERVAL_MIN  (default 10)
 * Redis:    REDIS_URL (optional)
 *
 * Degrades gracefully: if Redis is unavailable the lock step is skipped and
 * the sleep cycle still runs (single-instance safe).
 */

import { Redis } from "ioredis";
import { config } from "../config.js";
import type { Db } from "../db/client.js";
import { runSleep } from "../memory/sleep.js";

// ─── Lock store interface ─────────────────────────────────────────────────────
// Minimal surface that ioredis.Redis satisfies.
// Accepting an interface keeps tests and demo free of a live Redis dependency.

export interface LockStore {
  set(
    key: string,
    value: string,
    mode: "EX",
    ttlSeconds: number,
    condition: "NX",
  ): Promise<"OK" | null>;
  del(key: string): Promise<number>;
}

// ─── In-memory lock (used by demo/tests when Redis is unavailable) ────────────

export class MemoryLock implements LockStore {
  private readonly _held = new Map<string, number>(); // key → expiry epoch ms

  async set(
    key: string,
    _val: string,
    _m: "EX",
    ttl: number,
    _c: "NX",
  ): Promise<"OK" | null> {
    const expiry = this._held.get(key) ?? 0;
    if (Date.now() < expiry) return null; // still locked
    this._held.set(key, Date.now() + ttl * 1000);
    return "OK";
  }

  async del(key: string): Promise<number> {
    return this._held.delete(key) ? 1 : 0;
  }
}

// ─── Ring buffer ──────────────────────────────────────────────────────────────

export interface RunEntry {
  ts: number;            // epoch ms when the run started
  userId: string;
  consolidated: number;
  decayed: number;       // archived + expired
  inferred: number;
  durationMs: number;
  skipped: boolean;      // true = lock was already held by another instance
}

const RING_SIZE = 100;
const _ring: RunEntry[] = [];

function pushRing(entry: RunEntry): void {
  _ring.push(entry);
  if (_ring.length > RING_SIZE) _ring.shift();
}

/** Returns the run log in chronological order (oldest first). */
export function getRunLog(): ReadonlyArray<RunEntry> {
  return _ring;
}

// ─── Per-user run ─────────────────────────────────────────────────────────────

const LOCK_PREFIX = "engram:sleep_lock:";

/**
 * Try to acquire the per-user lock, run the sleep cycle, log the result.
 * Lock TTL is set to cover the full interval so a second instance skips.
 * On error the lock is released early so the next tick can retry.
 */
export async function runForUser(
  userId: string,
  db: Db,
  redis: LockStore | null,
  lockTtlSecs: number,
): Promise<RunEntry> {
  const entry: RunEntry = {
    ts: Date.now(),
    userId,
    consolidated: 0,
    decayed: 0,
    inferred: 0,
    durationMs: 0,
    skipped: false,
  };

  const lockKey = `${LOCK_PREFIX}${userId}`;

  // ── Acquire lock ────────────────────────────────────────────────────────────
  if (redis !== null) {
    let acquired = false;
    try {
      acquired = (await redis.set(lockKey, "1", "EX", lockTtlSecs, "NX")) === "OK";
    } catch {
      // Redis unavailable — proceed without lock (safe for single-instance mode)
    }
    if (!acquired) {
      entry.skipped = true;
      return entry;
    }
  }

  // ── Run the sleep cycle ─────────────────────────────────────────────────────
  const t0 = Date.now();
  try {
    const result = await runSleep(userId, db);
    entry.consolidated = result.consolidated_n;
    entry.decayed      = result.decayed_n;
    entry.inferred     = result.inferred_n;
    entry.skipped      = entry.skipped || result.skipped;
  } catch (err) {
    // Release lock early on failure so the next tick can retry sooner
    if (redis !== null) {
      try { await redis.del(lockKey); } catch { /* ignore */ }
    }
    throw err;
  }
  // On success: let the lock expire naturally (TTL prevents accidental overlap)

  entry.durationMs = Date.now() - t0;
  return entry;
}

// ─── Tick (one full sweep over all active users) ──────────────────────────────

function activeUserIds(db: Db): string[] {
  return (
    db
      .prepare("SELECT DISTINCT user_id FROM memories WHERE status = 'active'")
      .all() as { user_id: string }[]
  ).map((r) => r.user_id);
}

function fmtEntry(e: RunEntry): string {
  if (e.skipped) {
    return `[sleep] user=${e.userId} — skipped (already running elsewhere)`;
  }
  return (
    `[sleep] user=${e.userId}` +
    `  consolidated=${e.consolidated}` +
    `  decayed=${e.decayed}` +
    `  inferred=${e.inferred}` +
    `  duration=${e.durationMs}ms`
  );
}

/**
 * One sleep pass across all active users.
 * Runs users sequentially to limit peak DB / LLM pressure.
 */
export async function tick(
  db: Db,
  redis: LockStore | null,
  lockTtlSecs: number,
): Promise<RunEntry[]> {
  const users = activeUserIds(db);

  if (users.length === 0) {
    console.log("[scheduler] tick — no active users, nothing to do");
    return [];
  }

  const entries: RunEntry[] = [];

  for (const userId of users) {
    try {
      const entry = await runForUser(userId, db, redis, lockTtlSecs);
      pushRing(entry);
      entries.push(entry);
      console.log(fmtEntry(entry));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[scheduler] user=${userId} error: ${msg}`);
    }
  }

  return entries;
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────

let _timer: ReturnType<typeof setInterval> | null = null;
let _redis: Redis | null = null;

/**
 * Start the background scheduler.
 * Idempotent — calling twice is a no-op.
 * No-op unless ENGRAM_SCHEDULER=on.
 *
 * Connects to Redis with lazyConnect so the process doesn't crash if Redis
 * is temporarily unreachable on startup; errors are logged as warnings.
 */
export function startScheduler(db: Db): void {
  if (config.ENGRAM_SCHEDULER !== "on") return;
  if (_timer !== null) return; // already running

  const intervalMs  = config.ENGRAM_SLEEP_INTERVAL_MIN * 60 * 1000;
  const lockTtlSecs = Math.max(10, Math.floor(intervalMs / 1000 * 0.9));

  // Lock store: Redis SET NX when REDIS_URL is set, in-process otherwise.
  let store: LockStore;
  if (config.REDIS_URL) {
    _redis = new Redis(config.REDIS_URL, {
      lazyConnect: true,
      enableOfflineQueue: false,
      maxRetriesPerRequest: 0,
    });
    _redis.on("error", (err: Error) =>
      console.warn("[scheduler] Redis error (running without distributed lock):", err.message),
    );
    store = _redis as unknown as LockStore;
  } else {
    store = new MemoryLock();
  }

  console.log(
    `[scheduler] starting — interval=${config.ENGRAM_SLEEP_INTERVAL_MIN}m` +
    `  lockTtl=${lockTtlSecs}s  lock=${config.REDIS_URL ? "redis" : "in-process"}`,
  );

  const runTick = () =>
    tick(db, store, lockTtlSecs).catch((err) =>
      console.error("[scheduler] tick error:", err),
    );

  runTick(); // run immediately on start
  _timer = setInterval(runTick, intervalMs);
  _timer.unref(); // don't keep the process alive for the scheduler alone
}

export function stopScheduler(): void {
  if (_timer !== null) { clearInterval(_timer); _timer = null; }
  if (_redis !== null) { _redis.disconnect(); _redis = null; }
  console.log("[scheduler] stopped");
}
