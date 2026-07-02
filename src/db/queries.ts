/**
 * src/db/queries.ts
 *
 * Typed CRUD helpers for every table.
 * All writes are synchronous (better-sqlite3 is sync-first).
 * ULIDs are used for all primary keys.
 */

import { createHash } from "node:crypto";
import { ulid } from "ulid";
import type { Db } from "./client.js";

// ─── Memory type definitions ──────────────────────────────────────────────────

export type MemoryType =
  | "fact" | "preference" | "decision" | "event" | "skill" | "hypothesis" | "note";
export type MemoryStatus = "active" | "archived" | "superseded" | "expired";
export type MemorySource = "extracted" | "explicit" | "consolidated" | "inferred";
export type EdgeKind = "supersedes" | "derived_from" | "supports" | "contradicts";
export type BeliefReason = "confirmed" | "contradicted" | "decayed";

export interface Memory {
  id: string;
  user_id: string;
  type: MemoryType;
  content: string;
  content_hash: string;
  session_id?: string | null;
  embedding?: number[] | null;
  salience: number;
  confidence: number;
  source: MemorySource;
  pinned: number;           // 0 | 1
  status: MemoryStatus;
  superseded_by?: string | null;
  created_at: number;       // epoch ms
  last_accessed_at: number; // epoch ms
  access_count: number;
  ttl_at?: number | null;
  needs_review: number;     // 0 | 1 — belief floor hit on a fact/preference
  tags: string;             // JSON-encoded string[]
}

export interface NewMemory {
  user_id: string;
  type: MemoryType;
  content: string;
  embedding?: number[] | null;
  salience?: number;
  confidence?: number;
  source?: MemorySource;
  pinned?: number;
  ttl_at?: number | null;
  session_id?: string | null;
  tags?: string[];
}

export interface FtsResult {
  id: string;
  content: string;
  type: MemoryType;
  status: MemoryStatus;
  salience: number;
  confidence: number;
  pinned: number;
  tags: string;
  created_at: number;
  session_id: string | null;
  bm25: number;  // BM25 rank from FTS5 (more negative = better match)
}

export interface VecResult {
  id: string;
  content: string;
  type: MemoryType;
  status: MemoryStatus;
  salience: number;
  confidence: number;
  pinned: number;
  tags: string;
  created_at: number;
  session_id: string | null;
  distance: number;  // cosine distance (0 = identical)
}

// ─── Embedding helpers ────────────────────────────────────────────────────────

function embeddingToBuffer(embedding: number[]): Buffer {
  return Buffer.from(new Float32Array(embedding).buffer);
}

function bufferToEmbedding(buf: Buffer | null | undefined): number[] | null {
  if (!buf) return null;
  const f32 = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  return Array.from(f32);
}

// ─── Row → Memory mapper ──────────────────────────────────────────────────────

function rowToMemory(row: Record<string, unknown>): Memory {
  return {
    id: row.id as string,
    user_id: row.user_id as string,
    type: row.type as MemoryType,
    content: row.content as string,
    content_hash: row.content_hash as string,
    session_id: (row.session_id as string | null) ?? null,
    embedding: bufferToEmbedding(row.embedding as Buffer | null),
    salience: row.salience as number,
    confidence: row.confidence as number,
    source: row.source as MemorySource,
    pinned: row.pinned as number,
    status: row.status as MemoryStatus,
    superseded_by: (row.superseded_by as string | null) ?? null,
    created_at: row.created_at as number,
    last_accessed_at: row.last_accessed_at as number,
    access_count: row.access_count as number,
    ttl_at: (row.ttl_at as number | null) ?? null,
    needs_review: (row.needs_review as number) ?? 0,
    tags: row.tags as string,
  };
}

// ─── memories ────────────────────────────────────────────────────────────────

/** sha256 hex digest of memory content — the idempotency key for writes. */
export function contentHash(content: string): string {
  return createHash("sha256").update(content).digest("hex");
}

/**
 * Insert a memory (and its embedding into vec_memories) in a single transaction.
 * Idempotent: if an ACTIVE memory with the same (user_id, content_hash) already
 * exists, no new row is written — the existing memory is touched and its id
 * returned. Returns the memory id either way.
 */
export function insertMemory(db: Db, m: NewMemory): string {
  const id = ulid();
  const now = Date.now();
  const tags = JSON.stringify(m.tags ?? []);
  const hash = contentHash(m.content);
  const embBuf = m.embedding ? embeddingToBuffer(m.embedding) : null;

  const run = db.transaction((): string => {
    // Graceful duplicate handling: the partial unique index on
    // (user_id, content_hash) WHERE status='active' is the source of truth.
    const existing = db
      .prepare(
        "SELECT id FROM memories WHERE user_id = ? AND content_hash = ? AND status = 'active' LIMIT 1"
      )
      .get(m.user_id, hash) as { id: string } | undefined;
    if (existing) {
      touchMemory(db, existing.id);
      return existing.id;
    }

    db.prepare(`
      INSERT INTO memories
        (id, user_id, type, content, content_hash, embedding, salience, confidence,
         source, pinned, status, created_at, last_accessed_at, access_count,
         ttl_at, session_id, tags)
      VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, 0, ?, ?, ?)
    `).run(
      id,
      m.user_id,
      m.type,
      m.content,
      hash,
      embBuf,
      m.salience ?? 0.5,
      m.confidence ?? 0.8,
      m.source ?? "explicit",
      m.pinned ?? 0,
      now,
      now,
      m.ttl_at ?? null,
      m.session_id ?? null,
      tags,
    );

    if (embBuf) {
      db.prepare(
        "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)"
      ).run(id, embBuf);
    }
    return id;
  });

  return run();
}

/**
 * Find an active memory with exactly the same content for a user.
 * Used as the fast-path exact-duplicate check before embedding.
 */
export function findByExactContent(db: Db, userId: string, content: string): Memory | null {
  const row = db
    .prepare(
      "SELECT * FROM memories WHERE user_id = ? AND content = ? AND status = 'active' LIMIT 1"
    )
    .get(userId, content) as Record<string, unknown> | undefined;
  return row ? rowToMemory(row) : null;
}

/** Fetch a single memory by id. Returns null if not found. */
export function getMemory(db: Db, id: string): Memory | null {
  const row = db
    .prepare("SELECT * FROM memories WHERE id = ?")
    .get(id) as Record<string, unknown> | undefined;
  return row ? rowToMemory(row) : null;
}

/**
 * Change the status of a memory.
 * Optionally record which memory supersedes it.
 */
export function updateMemoryStatus(
  db: Db,
  id: string,
  status: MemoryStatus,
  superseded_by?: string
): void {
  db.prepare(`
    UPDATE memories SET status = ?, superseded_by = ? WHERE id = ?
  `).run(status, superseded_by ?? null, id);
}

/** Set a memory's confidence (caller clamps to [0,1] per config η rules). */
export function setConfidence(db: Db, id: string, confidence: number): void {
  db.prepare("UPDATE memories SET confidence = ? WHERE id = ?").run(confidence, id);
}

/** Flag/unflag a memory for human review (belief floor on fact/preference). */
export function setNeedsReview(db: Db, id: string, flag: boolean): void {
  db.prepare("UPDATE memories SET needs_review = ? WHERE id = ?").run(flag ? 1 : 0, id);
}

/** Bump last_accessed_at and increment access_count. */
export function touchMemory(db: Db, id: string): void {
  db.prepare(`
    UPDATE memories
    SET last_accessed_at = ?, access_count = access_count + 1
    WHERE id = ?
  `).run(Date.now(), id);
}

/** List memories with optional filtering. Ordered by last_accessed_at DESC. */
export interface ListFilter {
  user_id: string;
  type?: MemoryType;
  status?: MemoryStatus;
  pinned?: boolean;
  limit?: number;
  offset?: number;
}

export function listMemories(db: Db, filter: ListFilter): Memory[] {
  const conditions: string[] = ["user_id = ?"];
  const params: unknown[] = [filter.user_id];

  if (filter.type) { conditions.push("type = ?"); params.push(filter.type); }
  if (filter.status) { conditions.push("status = ?"); params.push(filter.status); }
  if (filter.pinned !== undefined) { conditions.push("pinned = ?"); params.push(filter.pinned ? 1 : 0); }

  params.push(filter.limit ?? 50);
  params.push(filter.offset ?? 0);

  const rows = db.prepare(`
    SELECT * FROM memories
    WHERE ${conditions.join(" AND ")}
    ORDER BY last_accessed_at DESC
    LIMIT ? OFFSET ?
  `).all(...params) as Record<string, unknown>[];

  return rows.map(rowToMemory);
}

// ─── FTS search ───────────────────────────────────────────────────────────────

/**
 * Full-text search via FTS5 (BM25 ranking).
 * Filters to the given user_id and status='active' (unless overridden).
 * Lower (more negative) rank = better match.
 */
export function ftsSearch(
  db: Db,
  userId: string,
  query: string,
  k = 10,
  status: MemoryStatus = "active"
): FtsResult[] {
  const rows = db.prepare(`
    SELECT
      m.id, m.content, m.type, m.status,
      m.salience, m.confidence, m.pinned, m.tags, m.created_at, m.session_id,
      f.rank AS bm25
    FROM memories_fts f
    JOIN memories m ON m.id = f.memory_id
    WHERE memories_fts MATCH ?
      AND m.user_id = ?
      AND m.status = ?
    ORDER BY f.rank
    LIMIT ?
  `).all(query, userId, status, k) as FtsResult[];

  return rows;
}

// ─── Vector (ANN) search ──────────────────────────────────────────────────────

/**
 * Approximate nearest-neighbour search via sqlite-vec (cosine distance).
 * Over-fetches by 3× to absorb user_id and status filtering after the KNN step.
 */
export function vecSearch(
  db: Db,
  userId: string,
  queryEmbedding: number[],
  k = 10,
  status: MemoryStatus = "active"
): VecResult[] {
  const queryBuf = embeddingToBuffer(queryEmbedding);
  const overFetch = k * 3;

  const rows = db.prepare(`
    SELECT
      m.id, m.content, m.type, m.status,
      m.salience, m.confidence, m.pinned, m.tags, m.created_at, m.session_id,
      v.distance
    FROM vec_memories v
    JOIN memories m ON m.id = v.memory_id
    WHERE v.embedding MATCH ? AND v.k = ?
      AND m.user_id = ?
      AND m.status = ?
    ORDER BY v.distance
    LIMIT ?
  `).all(queryBuf, overFetch, userId, status, k) as VecResult[];

  return rows;
}

// ─── memory_edges (the memory graph) ─────────────────────────────────────────

export interface MemoryEdge {
  id: string;
  src_id: string;
  dst_id: string;
  kind: EdgeKind;
  created_at: number;
}

/**
 * Add an edge to the memory graph. Idempotent: re-adding an existing
 * (src, dst, kind) triple returns the existing edge id.
 */
export function addEdge(db: Db, src_id: string, dst_id: string, kind: EdgeKind): string {
  const existing = db
    .prepare("SELECT id FROM memory_edges WHERE src_id = ? AND dst_id = ? AND kind = ?")
    .get(src_id, dst_id, kind) as { id: string } | undefined;
  if (existing) return existing.id;

  const id = ulid();
  db.prepare(`
    INSERT INTO memory_edges (id, src_id, dst_id, kind, created_at)
    VALUES (?, ?, ?, ?, ?)
  `).run(id, src_id, dst_id, kind, Date.now());
  return id;
}

/** All edges touching a memory, in either direction. */
export function edgesFor(db: Db, memoryId: string): MemoryEdge[] {
  return db.prepare(`
    SELECT * FROM memory_edges
    WHERE src_id = ? OR dst_id = ?
    ORDER BY created_at
  `).all(memoryId, memoryId) as MemoryEdge[];
}

// ─── belief_events (audit trail of learning) ─────────────────────────────────

export interface NewBeliefEvent {
  memory_id: string;
  delta: number;
  reason: BeliefReason;
  turn_id?: string | null;
}

export function insertBeliefEvent(db: Db, e: NewBeliefEvent): string {
  const id = ulid();
  db.prepare(`
    INSERT INTO belief_events (id, memory_id, delta, reason, turn_id, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(id, e.memory_id, e.delta, e.reason, e.turn_id ?? null, Date.now());
  return id;
}

// ─── sleep_runs ───────────────────────────────────────────────────────────────

export interface NewSleepRun {
  user_id: string;
  consolidated_n: number;
  decayed_n: number;
  inferred_n: number;
  notes?: string | null;
}

export function insertSleepRun(db: Db, r: NewSleepRun): string {
  const id = ulid();
  db.prepare(`
    INSERT INTO sleep_runs (id, user_id, run_at, consolidated_n, decayed_n, inferred_n, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `).run(id, r.user_id, Date.now(), r.consolidated_n, r.decayed_n, r.inferred_n, r.notes ?? null);
  return id;
}

// ─── episodic_log ─────────────────────────────────────────────────────────────

export interface EpisodicEntry {
  user_id: string;
  session_id: string;
  role: "user" | "assistant" | "system";
  content: string;
}

export function insertEpisodic(db: Db, entry: EpisodicEntry): string {
  const id = ulid();
  db.prepare(`
    INSERT INTO episodic_log (id, user_id, session_id, role, content, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(id, entry.user_id, entry.session_id, entry.role, entry.content, Date.now());
  return id;
}

export interface EpisodicRow extends EpisodicEntry {
  id: string;
  created_at: number;
}

/** Most recent N raw turns for a user (newest first) — sleep-cycle input. */
export function recentEpisodic(db: Db, userId: string, n: number): EpisodicRow[] {
  return db.prepare(`
    SELECT * FROM episodic_log
    WHERE user_id = ?
    ORDER BY created_at DESC
    LIMIT ?
  `).all(userId, n) as EpisodicRow[];
}

// ─── eval_runs ────────────────────────────────────────────────────────────────

export interface NewEvalRun {
  arm: "no_memory" | "full_history" | "engram";
  session_index: number;
  task_accuracy?: number | null;
  tokens_in_context?: number | null;
  recall_at_k?: number | null;
  forgetting_precision?: number | null;
  latency_p50_ms?: number | null;
  latency_p95_ms?: number | null;
  cost_usd?: number | null;
  metadata?: Record<string, unknown> | null;
}

export function insertEvalRun(db: Db, r: NewEvalRun): string {
  const id = ulid();
  db.prepare(`
    INSERT INTO eval_runs
      (id, run_at, arm, session_index, task_accuracy, tokens_in_context,
       recall_at_k, forgetting_precision, latency_p50_ms, latency_p95_ms,
       cost_usd, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    id, Date.now(), r.arm, r.session_index,
    r.task_accuracy ?? null, r.tokens_in_context ?? null,
    r.recall_at_k ?? null, r.forgetting_precision ?? null,
    r.latency_p50_ms ?? null, r.latency_p95_ms ?? null,
    r.cost_usd ?? null, r.metadata ? JSON.stringify(r.metadata) : null,
  );
  return id;
}

// ─── conflicts ────────────────────────────────────────────────────────────────

export interface ConflictEntry {
  new_id: string;
  old_id: string;
  resolution: "superseded" | "kept" | "merged";
  reasoning?: string;
}

export function insertConflict(db: Db, entry: ConflictEntry): string {
  const id = ulid();
  db.prepare(`
    INSERT INTO conflicts (id, new_id, old_id, resolution, reasoning, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(id, entry.new_id, entry.old_id, entry.resolution, entry.reasoning ?? null, Date.now());
  return id;
}

// ─── consolidations ───────────────────────────────────────────────────────────

export interface ConsolidationEntry {
  summary_id: string;
  source_ids: string[];
}

export function insertConsolidation(db: Db, entry: ConsolidationEntry): string {
  const id = ulid();
  db.prepare(`
    INSERT INTO consolidations (id, summary_id, source_ids, created_at)
    VALUES (?, ?, ?, ?)
  `).run(id, entry.summary_id, JSON.stringify(entry.source_ids), Date.now());
  return id;
}
