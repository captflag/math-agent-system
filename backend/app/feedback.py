"""Human-in-the-loop feedback: SQLite store + few-shot exemplar selection.

Highly-rated solutions become few-shot exemplars appended to the system prompt
(after the cached stable prefix, so they don't churn the prompt cache more than
necessary). This is the pragmatic core of what DSPy's BootstrapFewShot does,
without the framework dependency.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    final_answer TEXT NOT NULL,
    accuracy INTEGER NOT NULL,
    clarity INTEGER NOT NULL,
    comments TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def _conn(db_path: Path | None = None):
    path = db_path or settings.FEEDBACK_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_feedback(question: str, final_answer: str, accuracy: int,
                  clarity: int, comments: str | None = None,
                  db_path: Path | None = None) -> int:
    with _conn(db_path) as c:
        cur = c.execute(
            "INSERT INTO feedback (question, final_answer, accuracy, clarity, comments) "
            "VALUES (?, ?, ?, ?, ?)",
            (question, final_answer, accuracy, clarity, comments),
        )
        return cur.lastrowid


def get_stats(db_path: Path | None = None) -> dict:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT COUNT(*) AS n, AVG(accuracy) AS avg_accuracy, AVG(clarity) AS avg_clarity "
            "FROM feedback"
        ).fetchone()
        return {
            "count": row["n"],
            "avg_accuracy": round(row["avg_accuracy"], 2) if row["avg_accuracy"] else None,
            "avg_clarity": round(row["avg_clarity"], 2) if row["avg_clarity"] else None,
        }


def get_few_shot_examples(limit: int | None = None, db_path: Path | None = None) -> list[dict]:
    """Top-rated (accuracy=5, clarity>=4) Q/A pairs, newest first."""
    limit = limit or settings.FEW_SHOT_LIMIT
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT question, final_answer FROM feedback "
            "WHERE accuracy = 5 AND clarity >= 4 "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
