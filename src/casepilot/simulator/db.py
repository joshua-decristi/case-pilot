"""SQLite connection and schema helpers for the enterprise simulator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    symptom TEXT NOT NULL,
    application_fields TEXT NOT NULL,
    required_documents TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    doc_type TEXT NOT NULL,
    status TEXT NOT NULL,
    received_at TEXT,
    fields TEXT NOT NULL,
    content_summary TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    message TEXT NOT NULL,
    details TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS follow_up_tasks (
    task_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL UNIQUE,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_case_id ON documents(case_id);
CREATE INDEX IF NOT EXISTS idx_events_case_id ON events(case_id);
CREATE INDEX IF NOT EXISTS idx_follow_up_tasks_case_id ON follow_up_tasks(case_id);
"""

DROP_SQL = """
DROP TABLE IF EXISTS follow_up_tasks;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS documents;
DROP TABLE IF EXISTS cases;
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with FK enforcement and row factory.

    Per-request connections (see app dependency). ``check_same_thread=False``
    is required because FastAPI may open the dependency and run the endpoint
    on different worker threads. Timeout covers brief lock waits under writers.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def apply_schema(conn: sqlite3.Connection, *, reset: bool = False) -> None:
    """Create tables; when reset=True, drop existing tables first."""
    if reset:
        conn.executescript(DROP_SQL)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
