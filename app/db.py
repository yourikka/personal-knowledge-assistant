from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT UNIQUE NOT NULL,
                    source_type TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    title TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_links (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_id, target_id)
                );

                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_turns_session_id ON chat_turns(session_id, id DESC);
                """
            )

    def get_document_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE fingerprint = ?", (fingerprint,)).fetchone()
        return self._row_to_document(row) if row else None

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def upsert_document(self, record: dict[str, Any]) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, fingerprint, source_type, source_uri, title, raw_text, cleaned_text,
                    summary, category, confidence, tags_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_type = excluded.source_type,
                    source_uri = excluded.source_uri,
                    title = excluded.title,
                    raw_text = excluded.raw_text,
                    cleaned_text = excluded.cleaned_text,
                    summary = excluded.summary,
                    category = excluded.category,
                    confidence = excluded.confidence,
                    tags_json = excluded.tags_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record["id"],
                    record["fingerprint"],
                    record["source_type"],
                    record["source_uri"],
                    record["title"],
                    record["raw_text"],
                    record["cleaned_text"],
                    record["summary"],
                    record["category"],
                    record["confidence"],
                    json.dumps(record["tags"], ensure_ascii=False),
                    json.dumps(record["metadata"], ensure_ascii=False),
                    record.get("created_at", now),
                    now,
                ),
            )

    def list_documents(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_document(row) for row in rows]

    def iter_documents(self, exclude_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM documents"
        params: tuple[Any, ...] = ()
        if exclude_id:
            query += " WHERE id != ?"
            params = (exclude_id,)
        query += " ORDER BY created_at DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    def replace_links(self, source_id: str, related: list[dict[str, Any]]) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute("DELETE FROM document_links WHERE source_id = ?", (source_id,))
            for item in related:
                target_id = item["target_id"]
                score = float(item["score"])
                conn.execute(
                    """
                    INSERT OR REPLACE INTO document_links(source_id, target_id, score, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source_id, target_id, score, now),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO document_links(source_id, target_id, score, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (target_id, source_id, score, now),
                )

    def list_links(self, source_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT l.target_id, l.score, d.title, d.summary, d.source_uri
                FROM document_links l
                JOIN documents d ON d.id = l.target_id
                WHERE l.source_id = ?
                ORDER BY l.score DESC
                LIMIT ?
                """,
                (source_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_chat_turn(self, session_id: str, role: str, content: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO chat_turns(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, utc_now()),
            )

    def list_chat_turns(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def _row_to_document(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "fingerprint": row["fingerprint"],
            "source_type": row["source_type"],
            "source_uri": row["source_uri"],
            "title": row["title"],
            "raw_text": row["raw_text"],
            "cleaned_text": row["cleaned_text"],
            "summary": row["summary"],
            "category": row["category"],
            "confidence": row["confidence"],
            "tags": json.loads(row["tags_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

