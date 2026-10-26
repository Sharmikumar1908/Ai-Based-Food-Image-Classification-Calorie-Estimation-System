"""SQLite storage for aggregate session summaries only."""

import sqlite3
from pathlib import Path
from typing import Any

from utils import CLASS_NAMES, DATABASE_PATH


def _connect(db_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str | Path = DATABASE_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    columns = ",\n".join(f"{name} INTEGER NOT NULL DEFAULT 0" for name in CLASS_NAMES)
    with _connect(db_path) as connection:
        connection.execute(
            f"""CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                media_type TEXT NOT NULL,
                detected_subjects INTEGER NOT NULL,
                observations INTEGER NOT NULL,
                model_mode TEXT NOT NULL,
                {columns}
            )"""
        )


def add_session(summary: dict[str, Any], db_path: str | Path = DATABASE_PATH) -> int:
    column_names = ("media_type", "detected_subjects", "observations", "model_mode", *CLASS_NAMES)
    placeholders = ", ".join("?" for _ in column_names)
    values = [
        summary.get("media_type", "image"),
        int(summary.get("detected_subjects", 0)),
        int(summary.get("observations", 0)),
        summary.get("model_mode", "demo"),
        *(int(summary.get("counts", {}).get(name, 0)) for name in CLASS_NAMES),
    ]
    with _connect(db_path) as connection:
        cursor = connection.execute(
            f"INSERT INTO sessions ({', '.join(column_names)}) VALUES ({placeholders})", values
        )
        return int(cursor.lastrowid)


def list_sessions(db_path: str | Path = DATABASE_PATH) -> list[dict[str, Any]]:
    with _connect(db_path) as connection:
        rows = connection.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: int, db_path: str | Path = DATABASE_PATH) -> bool:
    with _connect(db_path) as connection:
        cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0