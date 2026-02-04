from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

DEFAULT_DB_PATH = os.getenv("CAUSALX_RESULTS_DB", "data/results.db")


@dataclass
class AnalysisRecord:
    analysis_id: str
    video_name: str
    created_at: str
    payload: dict[str, Any]


def _connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_results (
            analysis_id TEXT PRIMARY KEY,
            video_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    return conn


def save_result(analysis_id: str, video_name: str, payload: dict[str, Any]) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analysis_results (analysis_id, video_name, created_at, payload)
            VALUES (?, ?, datetime('now'), ?)
            """,
            (analysis_id, video_name, json.dumps(payload)),
        )
    conn.close()


def get_result(analysis_id: str) -> AnalysisRecord | None:
    conn = _connect()
    cur = conn.execute(
        """
        SELECT analysis_id, video_name, created_at, payload
        FROM analysis_results
        WHERE analysis_id = ?
        """,
        (analysis_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    payload = json.loads(row[3])
    return AnalysisRecord(
        analysis_id=row[0],
        video_name=row[1],
        created_at=row[2],
        payload=payload,
    )


def list_results(limit: int = 50) -> list[AnalysisRecord]:
    conn = _connect()
    cur = conn.execute(
        """
        SELECT analysis_id, video_name, created_at, payload
        FROM analysis_results
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    records: list[AnalysisRecord] = []
    for row in rows:
        records.append(
            AnalysisRecord(
                analysis_id=row[0],
                video_name=row[1],
                created_at=row[2],
                payload=json.loads(row[3]),
            )
        )
    return records
