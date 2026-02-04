from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

DEFAULT_DB_PATH = os.getenv("CAUSALX_RESULTS_DB", "data/results.db")


@dataclass
class LogRecord:
    log_id: int
    analysis_id: str
    event: str
    created_at: str
    metadata: dict[str, Any]


def _connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            event TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT NOT NULL
        )
        """
    )
    return conn


def log_event(analysis_id: str, event: str, metadata: dict[str, Any] | None = None) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            INSERT INTO analysis_logs (analysis_id, event, created_at, metadata)
            VALUES (?, ?, datetime('now'), ?)
            """,
            (analysis_id, event, json.dumps(metadata or {})),
        )
    conn.close()


def list_logs(analysis_id: str | None = None, limit: int = 200) -> list[LogRecord]:
    conn = _connect()
    if analysis_id:
        cur = conn.execute(
            """
            SELECT log_id, analysis_id, event, created_at, metadata
            FROM analysis_logs
            WHERE analysis_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (analysis_id, limit),
        )
    else:
        cur = conn.execute(
            """
            SELECT log_id, analysis_id, event, created_at, metadata
            FROM analysis_logs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    rows = cur.fetchall()
    conn.close()
    records: list[LogRecord] = []
    for row in rows:
        records.append(
            LogRecord(
                log_id=row[0],
                analysis_id=row[1],
                event=row[2],
                created_at=row[3],
                metadata=json.loads(row[4]),
            )
        )
    return records
