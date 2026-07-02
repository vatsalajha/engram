/**
 * src/agent/scheduler-demo.ts
 *
 * Shows two consecutive scheduler ticks with realistic data.
 * Uses an in-memory SQLite DB and an in-memory LockStore — no Redis required.
 *
 * Tick 1: finds stale events + active memories, runs decayAndExpire.
 * Tick 2: same users, same interval — lock is still held → all skipped.
 *
 * Usage:  tsx src/agent/scheduler-demo.ts
 */

import { openDb } from "../db/client.js";
import { insertMemory } from "../db/queries.js";
import { tick, getRunLog, MemoryLock } from "./scheduler.js";

// ─── Seed an in-memory DB ─────────────────────────────────────────────────────

const db = openDb(":memory:");
const lock = new MemoryLock();

// Lock TTL set to 30 s so tick-2 (fired immediately) always sees the lock held
const LOCK_TTL_SECS = 30;

const OLD = Date.now() - 400 * 86_400_000; // 400 days ago

// ── alice: one stale, never-accessed event + one fresh preference ─────────────
const staleEventId = insertMemory(db, {
  user_id: "alice",
  type: "event",
  content: "Alice attended a brief kick-off meeting.",
  salience: 0.05,
  confidence: 0.10,
});
db.prepare("UPDATE memories SET created_at = ?, last_accessed_at = ? WHERE id = ?")
  .run(OLD, OLD, staleEventId);

insertMemory(db, {
  user_id: "alice",
  type: "preference",
  content: "The user prefers dark mode across all applications.",
  salience: 0.90,
  confidence: 1.00,
});

// ── bob: one fresh fact — nothing to archive ──────────────────────────────────
insertMemory(db, {
  user_id: "bob",
  type: "fact",
  content: "Bob is a TypeScript engineer with five years of experience.",
  salience: 0.70,
  confidence: 0.95,
});

// ── carol: an archived memory past its TTL → will be expired ─────────────────
const archivedId = insertMemory(db, {
  user_id: "carol",
  type: "note",
  content: "Carol noted a temporary workaround for a login bug.",
  salience: 0.10,
  confidence: 0.50,
  ttl_at: Date.now() - 1000, // TTL already passed
});
db.prepare("UPDATE memories SET status = 'archived' WHERE id = ?").run(archivedId);

// Also give carol an active memory so she appears in activeUserIds
insertMemory(db, {
  user_id: "carol",
  type: "preference",
  content: "Carol prefers concise technical documentation.",
  salience: 0.65,
  confidence: 0.85,
});

// ─── Tick 1 ───────────────────────────────────────────────────────────────────

const sep = "─".repeat(64);

console.log(sep);
console.log(`TICK 1  ${new Date().toISOString()}`);
console.log(sep);
const t1 = await tick(db, lock, LOCK_TTL_SECS);

// ─── Tick 2 (lock still held for all users → every run skipped) ──────────────

console.log();
console.log(sep);
console.log(`TICK 2  ${new Date().toISOString()}  (same lock window → all skipped)`);
console.log(sep);
const t2 = await tick(db, lock, LOCK_TTL_SECS);

// ─── Ring buffer dump ─────────────────────────────────────────────────────────

console.log();
console.log(sep);
console.log("RING BUFFER  (all entries, chronological)");
console.log(sep);

const log = getRunLog();
for (const e of log) {
  const ts     = new Date(e.ts).toISOString();
  const status = e.skipped ? "SKIP" : "RAN ";
  const stats  = e.skipped
    ? "(lock held — no sleep performed)"
    : `consolidated=${e.consolidated}  decayed=${e.decayed}  inferred=${e.inferred}  duration=${e.durationMs}ms`;
  console.log(`  ${status}  ${ts}  user=${e.userId.padEnd(6)}  ${stats}`);
}

// ─── Verify acceptance criteria ───────────────────────────────────────────────

console.log();
console.log(sep);
console.log("ACCEPTANCE CHECK");
console.log(sep);

// Tick 1: all three users ran (not skipped)
console.assert(t1.every((e) => !e.skipped),  "tick-1: all runs should have acquired lock");
// Tick 2: all three users skipped (lock still held)
console.assert(t2.every((e) =>  e.skipped), "tick-2: all runs should have been skipped");

// alice: stale event should now be archived
const aliceMemories = db
  .prepare("SELECT id, content, status FROM memories WHERE user_id = 'alice'")
  .all() as { id: string; content: string; status: string }[];

const staleArchived = aliceMemories.find((m) => m.id === staleEventId);
console.assert(staleArchived?.status === "archived",
  `alice's stale event should be archived; got status=${staleArchived?.status}`);

// carol: TTL-expired archived memory should now be 'expired'
const carolArchived = db
  .prepare("SELECT status FROM memories WHERE id = ?")
  .get(archivedId) as { status: string } | undefined;
console.assert(carolArchived?.status === "expired",
  `carol's TTL-past memory should be expired; got status=${carolArchived?.status}`);

// No-op check: ring buffer has 6 entries (3 users × 2 ticks)
console.assert(log.length === 6, `ring buffer should have 6 entries, got ${log.length}`);

console.log("✓ tick-1 all acquired lock");
console.log("✓ tick-2 all skipped (lock held)");
console.log(`✓ alice's stale event: status=${staleArchived?.status}`);
console.log(`✓ carol's expired note: status=${carolArchived?.status}`);
console.log(`✓ ring buffer: ${log.length} entries`);
console.log(sep);
