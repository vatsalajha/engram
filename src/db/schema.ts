/**
 * src/db/schema.ts
 *
 * Idempotent schema creation (all IF NOT EXISTS).
 * Tables: memories, memories_fts (FTS5), vec_memories (sqlite-vec vec0),
 *         memory_edges, belief_events, episodic_log, sleep_runs, eval_runs,
 *         conflicts, consolidations (v1 — folded into memory_edges by the
 *         belief/sleep prompts, kept for the transition).
 * Triggers keep FTS5 in sync with memories on INSERT / UPDATE / DELETE.
 */

export const SCHEMA_SQL = /* sql */ `
  -- ─── Main memory store ────────────────────────────────────────────────────

  CREATE TABLE IF NOT EXISTS memories (
    id              TEXT    PRIMARY KEY,                       -- ULID
    user_id         TEXT    NOT NULL,
    type            TEXT    NOT NULL
                    CHECK(type IN ('fact','preference','decision','event','skill','hypothesis','note')),
    content         TEXT    NOT NULL,
    content_hash    TEXT    NOT NULL,                          -- sha256(content), idempotent writes
    embedding       BLOB,                                      -- float32[1024], canonical copy
    salience        REAL    NOT NULL DEFAULT 0.5
                    CHECK(salience BETWEEN 0 AND 1),
    confidence      REAL    NOT NULL DEFAULT 0.8
                    CHECK(confidence BETWEEN 0 AND 1),
    source          TEXT    NOT NULL DEFAULT 'explicit'
                    CHECK(source IN ('extracted','explicit','consolidated','inferred')),
    pinned          INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active','archived','superseded','expired')),
    superseded_by   TEXT    REFERENCES memories(id),
    created_at      INTEGER NOT NULL,                          -- epoch ms
    last_accessed_at INTEGER NOT NULL,                         -- epoch ms
    access_count    INTEGER NOT NULL DEFAULT 0,
    ttl_at          INTEGER,                                   -- hard expiry, epoch ms
    session_id      TEXT,                                      -- session provenance ("born in session N")
    needs_review    INTEGER NOT NULL DEFAULT 0,                -- belief floor hit on a fact/preference (§4.4)
    tags            TEXT    NOT NULL DEFAULT '[]'              -- JSON array of strings
  ) STRICT;

  -- Idempotent writes: one ACTIVE memory per (user, content hash). Partial so
  -- superseded/archived rows don't block re-learning the same content later.
  CREATE UNIQUE INDEX IF NOT EXISTS idx_mem_user_hash_active
    ON memories(user_id, content_hash) WHERE status = 'active';

  CREATE INDEX IF NOT EXISTS idx_mem_user_status
    ON memories(user_id, status);
  CREATE INDEX IF NOT EXISTS idx_mem_user_type_status
    ON memories(user_id, type, status);
  CREATE INDEX IF NOT EXISTS idx_mem_user_last_accessed
    ON memories(user_id, last_accessed_at);
  CREATE INDEX IF NOT EXISTS idx_mem_ttl
    ON memories(ttl_at) WHERE ttl_at IS NOT NULL;

  -- ─── FTS5 (Porter stemming) ───────────────────────────────────────────────
  -- memory_id is UNINDEXED (stored, not tokenised) so we can join back
  -- to memories without touching the rowid.

  CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    tags,
    tokenize = 'porter unicode61'
  );

  -- FTS sync: use regular DELETE for the existing row, then re-insert.
  -- (FTS5 'delete' command only works for external-content tables.)

  CREATE TRIGGER IF NOT EXISTS memories_fts_insert
    AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts(rowid, memory_id, content, tags)
      VALUES (new.rowid, new.id, new.content, new.tags);
    END;

  CREATE TRIGGER IF NOT EXISTS memories_fts_delete
    AFTER DELETE ON memories BEGIN
      DELETE FROM memories_fts WHERE rowid = old.rowid;
    END;

  CREATE TRIGGER IF NOT EXISTS memories_fts_update
    AFTER UPDATE ON memories BEGIN
      DELETE FROM memories_fts WHERE rowid = old.rowid;
      INSERT INTO memories_fts(rowid, memory_id, content, tags)
      VALUES (new.rowid, new.id, new.content, new.tags);
    END;

  -- ─── sqlite-vec ANN index (cosine, 1024-dim float) ───────────────────────

  -- distance_metric=cosine so dedup threshold maps directly to 1 - cosine_similarity
  CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    memory_id text primary key,
    embedding float[1024] distance_metric=cosine
  );

  -- ─── Memory graph (edges between memories) ───────────────────────────────

  CREATE TABLE IF NOT EXISTS memory_edges (
    id         TEXT    PRIMARY KEY,
    src_id     TEXT    NOT NULL REFERENCES memories(id),
    dst_id     TEXT    NOT NULL REFERENCES memories(id),
    kind       TEXT    NOT NULL
               CHECK(kind IN ('supersedes','derived_from','supports','contradicts')),
    created_at INTEGER NOT NULL
  ) STRICT;

  CREATE INDEX IF NOT EXISTS idx_edges_src ON memory_edges(src_id);
  CREATE INDEX IF NOT EXISTS idx_edges_dst ON memory_edges(dst_id);
  CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
    ON memory_edges(src_id, dst_id, kind);

  -- ─── Belief events (audit trail of confidence learning) ──────────────────

  CREATE TABLE IF NOT EXISTS belief_events (
    id         TEXT    PRIMARY KEY,
    memory_id  TEXT    NOT NULL REFERENCES memories(id),
    delta      REAL    NOT NULL,                               -- signed confidence change
    reason     TEXT    NOT NULL
               CHECK(reason IN ('confirmed','contradicted','decayed')),
    turn_id    TEXT,                                           -- provenance: which turn caused it
    created_at INTEGER NOT NULL
  ) STRICT;

  CREATE INDEX IF NOT EXISTS idx_belief_memory ON belief_events(memory_id);

  -- ─── Sleep runs (consolidate · decay · infer audit log) ──────────────────

  CREATE TABLE IF NOT EXISTS sleep_runs (
    id             TEXT    PRIMARY KEY,
    user_id        TEXT    NOT NULL,
    run_at         INTEGER NOT NULL,
    consolidated_n INTEGER NOT NULL DEFAULT 0,
    decayed_n      INTEGER NOT NULL DEFAULT 0,
    inferred_n     INTEGER NOT NULL DEFAULT 0,
    notes          TEXT
  ) STRICT;

  CREATE INDEX IF NOT EXISTS idx_sleep_user_ts ON sleep_runs(user_id, run_at);

  -- ─── Episodic log (raw turns before extraction) ──────────────────────────

  CREATE TABLE IF NOT EXISTS episodic_log (
    id         TEXT    PRIMARY KEY,
    user_id    TEXT    NOT NULL,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
    content    TEXT    NOT NULL,
    created_at INTEGER NOT NULL
  ) STRICT;

  CREATE INDEX IF NOT EXISTS idx_episodic_user_session
    ON episodic_log(user_id, session_id);
  CREATE INDEX IF NOT EXISTS idx_episodic_user_ts
    ON episodic_log(user_id, created_at);

  -- ─── Conflicts (detected contradictions between memories) ────────────────

  CREATE TABLE IF NOT EXISTS conflicts (
    id         TEXT    PRIMARY KEY,
    new_id     TEXT    NOT NULL REFERENCES memories(id),
    old_id     TEXT    NOT NULL REFERENCES memories(id),
    resolution TEXT    NOT NULL CHECK(resolution IN ('superseded','kept','merged')),
    reasoning  TEXT,
    created_at INTEGER NOT NULL
  ) STRICT;

  CREATE INDEX IF NOT EXISTS idx_conflicts_new ON conflicts(new_id);
  CREATE INDEX IF NOT EXISTS idx_conflicts_old ON conflicts(old_id);

  -- ─── Consolidations (episodic → semantic merges) ─────────────────────────

  CREATE TABLE IF NOT EXISTS consolidations (
    id         TEXT    PRIMARY KEY,
    summary_id TEXT    NOT NULL REFERENCES memories(id),
    source_ids TEXT    NOT NULL,   -- JSON array of memory ids
    created_at INTEGER NOT NULL
  ) STRICT;

  CREATE INDEX IF NOT EXISTS idx_consolidations_summary
    ON consolidations(summary_id);

  -- ─── Eval runs (benchmark results) ──────────────────────────────────────

  CREATE TABLE IF NOT EXISTS eval_runs (
    id                  TEXT    PRIMARY KEY,
    run_at              INTEGER NOT NULL,
    arm                 TEXT    NOT NULL
                        CHECK(arm IN ('no_memory','full_history','engram')),
    session_index       INTEGER NOT NULL,
    task_accuracy       REAL,
    tokens_in_context   INTEGER,
    recall_at_k         REAL,
    forgetting_precision REAL,
    latency_p50_ms      REAL,
    latency_p95_ms      REAL,
    cost_usd            REAL,
    metadata            TEXT    -- JSON
  ) STRICT;

  CREATE INDEX IF NOT EXISTS idx_eval_arm_session
    ON eval_runs(arm, session_index);
`;
