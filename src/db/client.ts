/**
 * src/db/client.ts
 *
 * better-sqlite3 database factory + process-level singleton.
 * - WAL mode for concurrent reads
 * - sqlite-vec extension for vec0 ANN queries
 * - Schema applied once at open (idempotent IF NOT EXISTS)
 *
 * Usage:
 *   Production — call initDb(config.ENGRAM_DB_PATH) at startup, then getDb() anywhere.
 *   Tests      — call openDb(':memory:') directly; no config dependency.
 */

import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";
import { SCHEMA_SQL } from "./schema.js";

export type Db = Database.Database;

/**
 * Open (or create) a SQLite database, apply all pragmas + schema, and return it.
 * Safe to call with ':memory:' for isolated in-process test databases.
 */
export function openDb(path: string): Db {
  const db = new Database(path);

  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  db.pragma("busy_timeout = 5000");

  // Load sqlite-vec so vec0 virtual tables are available
  loadVec(db);

  // Apply schema — all statements use IF NOT EXISTS, safe to re-run
  db.exec(SCHEMA_SQL);

  return db;
}

// ─── Process-level singleton ──────────────────────────────────────────────────

let _db: Db | null = null;

/**
 * Initialize the process-level singleton. Must be called once at startup
 * before getDb() is used (typically in src/api/server.ts).
 */
export function initDb(path: string): Db {
  if (!_db) {
    _db = openDb(path);
  }
  return _db;
}

/**
 * Return the already-initialized singleton. Throws if initDb() was never called.
 */
export function getDb(): Db {
  if (!_db) throw new Error("Database not initialized — call initDb(path) first.");
  return _db;
}

/** Close and reset the singleton (useful in tests). */
export function closeDb(): void {
  _db?.close();
  _db = null;
}
