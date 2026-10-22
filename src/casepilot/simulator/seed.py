"""Deterministic fixture seed for the enterprise simulator SQLite database.

Destructive reseed requires an explicit ``reset=True`` / ``--reset``. The FastAPI
app never auto-seeds or overwrites the database on startup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from casepilot.shared import get_settings
from casepilot.simulator.db import apply_schema, connect

# Repo root: src/casepilot/simulator/seed.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES = _REPO_ROOT / "fixtures" / "enterprise" / "cases.json"


def _load_fixture(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"fixture root must be an object: {path}")
    for key in ("cases", "documents", "events"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"fixture missing list '{key}': {path}")
    return data


def _insert_records(conn: Any, data: dict[str, Any]) -> None:
    """Map fixture shapes onto contract-aligned SQLite columns and insert."""
    generated_at = data.get("generated_at") or "2024-06-01T00:00:00Z"

    for case in data["cases"]:
        conn.execute(
            """
            INSERT INTO cases (
                case_id, status, symptom, application_fields,
                required_documents, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case["id"],
                case["status"],
                case["symptom"],
                json.dumps(case.get("application") or {}),
                json.dumps(case.get("required_documents") or []),
                generated_at,
                None,
            ),
        )

    for doc in data["documents"]:
        conn.execute(
            """
            INSERT INTO documents (
                document_id, case_id, doc_type, status,
                received_at, fields, content_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc["id"],
                doc["case_id"],
                doc["document_type"],
                doc["status"],
                None,
                json.dumps(doc.get("fields") or {}),
                doc.get("content"),
            ),
        )

    for event in data["events"]:
        # Fixture `status` becomes ProcessingEvent.message; details stay as-is.
        conn.execute(
            """
            INSERT INTO events (
                event_id, case_id, event_type, timestamp, message, details
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event["id"],
                event["case_id"],
                event["event_type"],
                event["created_at"],
                str(event.get("status") or ""),
                json.dumps(event.get("details") or {}),
            ),
        )


def seed(
    db_path: Path | None = None,
    fixtures_path: Path | None = None,
    *,
    reset: bool = False,
) -> int:
    """Load enterprise JSON fixtures into SQLite.

    Default ``reset=False`` is non-destructive: refuses to overwrite a non-empty
    DB. Pass ``reset=True`` or CLI ``--reset`` to drop tables and reseed.
    The FastAPI app never calls this on startup.
    """
    if db_path is None:
        path = get_settings().simulator_db_path
    else:
        path = Path(db_path)
    fixtures = Path(fixtures_path) if fixtures_path is not None else DEFAULT_FIXTURES

    if not fixtures.is_file():
        print(f"Fixtures not found: {fixtures}", file=sys.stderr)
        return 1

    data = _load_fixture(fixtures)
    conn = connect(path)
    try:
        if not reset:
            apply_schema(conn, reset=False)
            row = conn.execute("SELECT COUNT(*) AS n FROM cases").fetchone()
            if row is not None and int(row["n"]) > 0:
                print(
                    f"Refusing to overwrite non-empty simulator DB at {path}. "
                    "Pass reset=True or --reset for a destructive reseed.",
                    file=sys.stderr,
                )
                return 1
        apply_schema(conn, reset=reset)
        try:
            if not reset:
                for table in ("follow_up_tasks", "events", "documents", "cases"):
                    conn.execute(f"DELETE FROM {table}")
            _insert_records(conn, data)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    print(
        f"Seeded {len(data['cases'])} cases, "
        f"{len(data['documents'])} documents, "
        f"{len(data['events'])} events → {path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed the CasePilot enterprise simulator SQLite database."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Destructively drop and recreate tables before loading fixtures.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Target SQLite path (default: settings.simulator_db_path).",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=None,
        help=f"Fixture JSON path (default: {DEFAULT_FIXTURES}).",
    )
    args = parser.parse_args(argv)
    return seed(db_path=args.db_path, fixtures_path=args.fixtures, reset=args.reset)


if __name__ == "__main__":
    raise SystemExit(main())
