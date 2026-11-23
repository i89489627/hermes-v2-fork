-- hermes-v2 trace store schema (plan P11.T1).
--
-- Idempotent: every statement uses CREATE ... IF NOT EXISTS, so TraceStore
-- may run this script on every (re)connection. Statement order matters: the
-- FTS5 virtual table is created LAST because sqlite3's executescript() stops
-- at the first failing statement — if the runtime SQLite lacks the fts5
-- module, TraceStore degrades to LIKE search while keeping the base table
-- and indexes.

CREATE TABLE IF NOT EXISTS traces (
    trace_id      TEXT PRIMARY KEY,
    turn_id       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    day           TEXT NOT NULL,
    user_prompt   TEXT NOT NULL,
    models_used   TEXT NOT NULL DEFAULT '[]',
    payload       TEXT NOT NULL DEFAULT '{}',
    quality_label TEXT,
    pii_flagged   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_quality ON traces(quality_label);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);

CREATE VIRTUAL TABLE IF NOT EXISTS traces_fts USING fts5(
    trace_id UNINDEXED,
    user_prompt,
    payload
);
