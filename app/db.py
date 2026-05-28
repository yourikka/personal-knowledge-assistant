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
        conn.execute("PRAGMA foreign_keys = ON")
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

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS document_sections (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    section_index INTEGER NOT NULL,
                    heading TEXT NOT NULL,
                    heading_path_json TEXT NOT NULL,
                    text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    scope TEXT NOT NULL DEFAULT 'session',
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    ttl_seconds INTEGER,
                    last_accessed_at TEXT,
                    conflict_key TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_document_id TEXT NOT NULL,
                    evidence_chunk_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(source_entity_id) REFERENCES graph_entities(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_entity_id) REFERENCES graph_entities(id) ON DELETE CASCADE,
                    FOREIGN KEY(evidence_document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY(evidence_chunk_id) REFERENCES document_chunks(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS document_entities (
                    document_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    mention_count INTEGER NOT NULL,
                    first_chunk_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (document_id, entity_id),
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY(entity_id) REFERENCES graph_entities(id) ON DELETE CASCADE,
                    FOREIGN KEY(first_chunk_id) REFERENCES document_chunks(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_document_sections_document_id ON document_sections(document_id, section_index);
                CREATE INDEX IF NOT EXISTS idx_chat_turns_session_id ON chat_turns(session_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_records_session_id ON memory_records(session_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_records_kind ON memory_records(kind, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_records_scope ON memory_records(scope, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_records_conflict_key ON memory_records(conflict_key, status);
                CREATE INDEX IF NOT EXISTS idx_graph_entities_name ON graph_entities(name, entity_type);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_entity_id, relation);
                CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_entity_id, relation);
                CREATE INDEX IF NOT EXISTS idx_document_entities_entity ON document_entities(entity_id, document_id);
                """
            )
            self._ensure_memory_columns(conn)

    def _ensure_memory_columns(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(memory_records)").fetchall()}
        migrations = {
            "scope": "ALTER TABLE memory_records ADD COLUMN scope TEXT NOT NULL DEFAULT 'session'",
            "ttl_seconds": "ALTER TABLE memory_records ADD COLUMN ttl_seconds INTEGER",
            "last_accessed_at": "ALTER TABLE memory_records ADD COLUMN last_accessed_at TEXT",
            "conflict_key": "ALTER TABLE memory_records ADD COLUMN conflict_key TEXT",
            "status": "ALTER TABLE memory_records ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
        }
        for column, statement in migrations.items():
            if column not in columns:
                conn.execute(statement)

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

    def delete_document(self, document_id: str) -> bool:
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone()
            if not exists:
                return False
            conn.execute("DELETE FROM document_links WHERE source_id = ? OR target_id = ?", (document_id, document_id))
            conn.execute("DELETE FROM graph_edges WHERE evidence_document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_entities WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_sections WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return True

    def search_documents_keyword(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        terms = [term.strip().lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        rows_by_id: dict[str, dict[str, Any]] = {}
        with self.connect() as conn:
            for term in terms:
                like = f"%{term}%"
                rows = conn.execute(
                    """
                    SELECT *
                    FROM documents
                    WHERE lower(title) LIKE ?
                       OR lower(summary) LIKE ?
                       OR lower(cleaned_text) LIKE ?
                       OR lower(tags_json) LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (like, like, like, like, limit),
                ).fetchall()
                for row in rows:
                    rows_by_id[row["id"]] = self._row_to_document(row)

        return list(rows_by_id.values())[:limit]

    def replace_document_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO document_chunks(
                        id, document_id, chunk_index, text, char_start, char_end, metadata_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        document_id,
                        int(chunk["chunk_index"]),
                        chunk["text"],
                        int(chunk["char_start"]),
                        int(chunk["char_end"]),
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                        now,
                    ),
                )

    def list_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def replace_document_sections(self, document_id: str, sections: list[dict[str, Any]]) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute("DELETE FROM document_sections WHERE document_id = ?", (document_id,))
            for section in sections:
                conn.execute(
                    """
                    INSERT INTO document_sections(
                        id, document_id, section_index, heading, heading_path_json,
                        text, char_start, char_end, metadata_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        section["id"],
                        document_id,
                        int(section["section_index"]),
                        section["heading"],
                        json.dumps(section.get("heading_path", []), ensure_ascii=False),
                        section["text"],
                        int(section["char_start"]),
                        int(section["char_end"]),
                        json.dumps(section.get("metadata", {}), ensure_ascii=False),
                        now,
                    ),
                )

    def list_document_sections(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM document_sections
                WHERE document_id = ?
                ORDER BY section_index ASC
                """,
                (document_id,),
            ).fetchall()
        return [self._row_to_section(row) for row in rows]

    def get_section(self, section_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM document_sections WHERE id = ?", (section_id,)).fetchone()
        return self._row_to_section(row) if row else None

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM document_chunks WHERE id = ?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def iter_chunks(self, exclude_document_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM document_chunks"
        params: tuple[Any, ...] = ()
        if exclude_document_id:
            query += " WHERE document_id != ?"
            params = (exclude_document_id,)
        query += " ORDER BY document_id, chunk_index ASC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def search_chunks_keyword(
        self,
        query: str,
        limit: int = 20,
        exclude_document_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        terms = [term.strip().lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        exclude_document_ids = exclude_document_ids or set()
        rows_by_id: dict[str, dict[str, Any]] = {}
        with self.connect() as conn:
            for term in terms:
                like = f"%{term}%"
                rows = conn.execute(
                    """
                    SELECT c.*
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE lower(c.text) LIKE ?
                       OR lower(d.title) LIKE ?
                       OR lower(d.summary) LIKE ?
                       OR lower(d.tags_json) LIKE ?
                    ORDER BY d.created_at DESC, c.chunk_index ASC
                    LIMIT ?
                    """,
                    (like, like, like, like, limit),
                ).fetchall()
                for row in rows:
                    if row["document_id"] in exclude_document_ids:
                        continue
                    rows_by_id[row["id"]] = self._row_to_chunk(row)

        return list(rows_by_id.values())[:limit]

    def search_sections_keyword(
        self,
        query: str,
        limit: int = 20,
        exclude_document_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        terms = [term.strip().lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        exclude_document_ids = exclude_document_ids or set()
        rows_by_id: dict[str, dict[str, Any]] = {}
        with self.connect() as conn:
            for term in terms:
                like = f"%{term}%"
                rows = conn.execute(
                    """
                    SELECT s.*
                    FROM document_sections s
                    JOIN documents d ON d.id = s.document_id
                    WHERE lower(s.heading) LIKE ?
                       OR lower(s.text) LIKE ?
                       OR lower(d.title) LIKE ?
                       OR lower(d.summary) LIKE ?
                       OR lower(d.tags_json) LIKE ?
                    ORDER BY d.created_at DESC, s.section_index ASC
                    LIMIT ?
                    """,
                    (like, like, like, like, like, limit),
                ).fetchall()
                for row in rows:
                    if row["document_id"] in exclude_document_ids:
                        continue
                    section = self._row_to_section(row)
                    rows_by_id[section["id"]] = section

        return list(rows_by_id.values())[:limit]

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

    def upsert_memory(self, record: dict[str, Any]) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_records(
                    id, session_id, scope, kind, content, importance, tags_json, metadata_json,
                    ttl_seconds, last_accessed_at, conflict_key, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    scope = excluded.scope,
                    kind = excluded.kind,
                    content = excluded.content,
                    importance = excluded.importance,
                    tags_json = excluded.tags_json,
                    metadata_json = excluded.metadata_json,
                    ttl_seconds = excluded.ttl_seconds,
                    conflict_key = excluded.conflict_key,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    record["id"],
                    record.get("session_id"),
                    record.get("scope", "session"),
                    record["kind"],
                    record["content"],
                    float(record.get("importance", 0.5)),
                    json.dumps(record.get("tags", []), ensure_ascii=False),
                    json.dumps(record.get("metadata", {}), ensure_ascii=False),
                    record.get("ttl_seconds"),
                    record.get("last_accessed_at"),
                    record.get("conflict_key"),
                    record.get("status", "active"),
                    record.get("created_at", now),
                    now,
                ),
            )

    def supersede_conflicting_memories(self, conflict_key: str, keep_id: str | None = None) -> int:
        if not conflict_key:
            return 0
        now = utc_now()
        with self.connect() as conn:
            if keep_id:
                cursor = conn.execute(
                    """
                    UPDATE memory_records
                    SET status = 'superseded', updated_at = ?
                    WHERE conflict_key = ? AND id != ? AND status = 'active'
                    """,
                    (now, conflict_key, keep_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE memory_records
                    SET status = 'superseded', updated_at = ?
                    WHERE conflict_key = ? AND status = 'active'
                    """,
                    (now, conflict_key),
                )
        return int(cursor.rowcount)

    def touch_memory_access(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = utc_now()
        placeholders = ",".join("?" for _ in memory_ids)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE memory_records SET last_accessed_at = ? WHERE id IN ({placeholders})",
                (now, *memory_ids),
            )

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM memory_records WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(
        self,
        session_id: str | None = None,
        limit: int = 20,
        include_global: bool = True,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if session_id and include_global:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM memory_records
                    WHERE (session_id = ? OR session_id IS NULL)
                      AND status = 'active'
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            elif session_id:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM memory_records
                    WHERE session_id = ?
                      AND status = 'active'
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM memory_records
                    WHERE status = 'active'
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_memories_keyword(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 20,
        include_global: bool = True,
    ) -> list[dict[str, Any]]:
        terms = [term.strip().lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        rows_by_id: dict[str, dict[str, Any]] = {}
        with self.connect() as conn:
            for term in terms:
                like = f"%{term}%"
                if session_id and include_global:
                    rows = conn.execute(
                        """
                        SELECT *
                        FROM memory_records
                        WHERE (session_id = ? OR session_id IS NULL)
                          AND status = 'active'
                          AND (lower(content) LIKE ? OR lower(tags_json) LIKE ?)
                        ORDER BY importance DESC, updated_at DESC
                        LIMIT ?
                        """,
                        (session_id, like, like, limit),
                    ).fetchall()
                elif session_id:
                    rows = conn.execute(
                        """
                        SELECT *
                        FROM memory_records
                        WHERE session_id = ?
                          AND status = 'active'
                          AND (lower(content) LIKE ? OR lower(tags_json) LIKE ?)
                        ORDER BY importance DESC, updated_at DESC
                        LIMIT ?
                        """,
                        (session_id, like, like, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT *
                        FROM memory_records
                        WHERE status = 'active'
                          AND (lower(content) LIKE ? OR lower(tags_json) LIKE ?)
                        ORDER BY importance DESC, updated_at DESC
                        LIMIT ?
                        """,
                        (like, like, limit),
                    ).fetchall()
                for row in rows:
                    memory = self._row_to_memory(row)
                    rows_by_id[memory["id"]] = memory
        return list(rows_by_id.values())[:limit]

    def delete_memory(self, memory_id: str) -> bool:
        with self.connect() as conn:
            exists = conn.execute("SELECT 1 FROM memory_records WHERE id = ?", (memory_id,)).fetchone()
            if not exists:
                return False
            conn.execute("DELETE FROM memory_records WHERE id = ?", (memory_id,))
        return True

    def replace_document_graph(
        self,
        document_id: str,
        entities: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute("DELETE FROM graph_edges WHERE evidence_document_id = ?", (document_id,))
            conn.execute("DELETE FROM document_entities WHERE document_id = ?", (document_id,))
            for entity in entities:
                conn.execute(
                    """
                    INSERT INTO graph_entities(
                        id, name, entity_type, aliases_json, metadata_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        entity_type = excluded.entity_type,
                        aliases_json = excluded.aliases_json,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        entity["id"],
                        entity["name"],
                        entity["entity_type"],
                        json.dumps(entity.get("aliases", []), ensure_ascii=False),
                        json.dumps(entity.get("metadata", {}), ensure_ascii=False),
                        entity.get("created_at", now),
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO document_entities(
                        document_id, entity_id, mention_count, first_chunk_id, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id, entity_id) DO UPDATE SET
                        mention_count = excluded.mention_count,
                        first_chunk_id = excluded.first_chunk_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        document_id,
                        entity["id"],
                        int(entity.get("mention_count", 1)),
                        entity.get("first_chunk_id"),
                        now,
                        now,
                    ),
                )
            for edge in edges:
                conn.execute(
                    """
                    INSERT INTO graph_edges(
                        id, source_entity_id, target_entity_id, relation, confidence,
                        evidence_document_id, evidence_chunk_id, metadata_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        relation = excluded.relation,
                        confidence = excluded.confidence,
                        evidence_document_id = excluded.evidence_document_id,
                        evidence_chunk_id = excluded.evidence_chunk_id,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        edge["id"],
                        edge["source_entity_id"],
                        edge["target_entity_id"],
                        edge["relation"],
                        float(edge.get("confidence", 0.5)),
                        document_id,
                        edge.get("evidence_chunk_id"),
                        json.dumps(edge.get("metadata", {}), ensure_ascii=False),
                        edge.get("created_at", now),
                        now,
                    ),
                )

    def list_document_entities(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT e.*, de.mention_count, de.first_chunk_id
                FROM document_entities de
                JOIN graph_entities e ON e.id = de.entity_id
                WHERE de.document_id = ?
                ORDER BY de.mention_count DESC, e.name ASC
                """,
                (document_id,),
            ).fetchall()
        return [self._row_to_graph_entity(row) for row in rows]

    def list_document_graph_edges(self, document_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT ge.*, src.name AS source_name, src.entity_type AS source_type,
                       dst.name AS target_name, dst.entity_type AS target_type
                FROM graph_edges ge
                JOIN graph_entities src ON src.id = ge.source_entity_id
                JOIN graph_entities dst ON dst.id = ge.target_entity_id
                WHERE ge.evidence_document_id = ?
                ORDER BY ge.confidence DESC, ge.updated_at DESC
                LIMIT ?
                """,
                (document_id, limit),
            ).fetchall()
        return [self._row_to_graph_edge(row) for row in rows]

    def find_entities_by_names(self, names: list[str], limit: int = 20) -> list[dict[str, Any]]:
        normalized = [name.strip().lower() for name in names if name.strip()]
        if not normalized:
            return []
        rows_by_id: dict[str, dict[str, Any]] = {}
        with self.connect() as conn:
            for name in normalized:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM graph_entities
                    WHERE lower(name) = ?
                       OR lower(aliases_json) LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (name, f"%{name}%", limit),
                ).fetchall()
                for row in rows:
                    entity = self._row_to_graph_entity(row)
                    rows_by_id[entity["id"]] = entity
        return list(rows_by_id.values())[:limit]

    def graph_documents_for_entities(self, entity_ids: list[str], limit: int = 10) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT d.*, de.entity_id, de.mention_count, e.name AS entity_name, e.entity_type
                FROM document_entities de
                JOIN documents d ON d.id = de.document_id
                JOIN graph_entities e ON e.id = de.entity_id
                WHERE de.entity_id IN ({placeholders})
                ORDER BY de.mention_count DESC, d.updated_at DESC
                LIMIT ?
                """,
                (*entity_ids, limit),
            ).fetchall()
        return [self._row_to_graph_document(row) for row in rows]

    def graph_neighbors(self, entity_ids: list[str], limit: int = 20) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT ge.*, src.name AS source_name, src.entity_type AS source_type,
                       dst.name AS target_name, dst.entity_type AS target_type
                FROM graph_edges ge
                JOIN graph_entities src ON src.id = ge.source_entity_id
                JOIN graph_entities dst ON dst.id = ge.target_entity_id
                WHERE ge.source_entity_id IN ({placeholders})
                   OR ge.target_entity_id IN ({placeholders})
                ORDER BY ge.confidence DESC, ge.updated_at DESC
                LIMIT ?
                """,
                (*entity_ids, *entity_ids, limit),
            ).fetchall()
        return [self._row_to_graph_edge(row) for row in rows]

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

    def _row_to_chunk(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "text": row["text"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    def _row_to_section(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "section_index": row["section_index"],
            "heading": row["heading"],
            "heading_path": json.loads(row["heading_path_json"]),
            "text": row["text"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    def _row_to_memory(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "scope": row["scope"],
            "kind": row["kind"],
            "content": row["content"],
            "importance": row["importance"],
            "tags": json.loads(row["tags_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "ttl_seconds": row["ttl_seconds"],
            "last_accessed_at": row["last_accessed_at"],
            "conflict_key": row["conflict_key"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_graph_entity(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "aliases": json.loads(row["aliases_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "mention_count": row["mention_count"] if "mention_count" in row.keys() else None,
            "first_chunk_id": row["first_chunk_id"] if "first_chunk_id" in row.keys() else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_graph_edge(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "source_entity_id": row["source_entity_id"],
            "target_entity_id": row["target_entity_id"],
            "source_name": row["source_name"],
            "source_type": row["source_type"],
            "target_name": row["target_name"],
            "target_type": row["target_type"],
            "relation": row["relation"],
            "confidence": row["confidence"],
            "evidence_document_id": row["evidence_document_id"],
            "evidence_chunk_id": row["evidence_chunk_id"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_graph_document(self, row: sqlite3.Row) -> dict[str, Any]:
        document = self._row_to_document(row)
        document["graph_entity_id"] = row["entity_id"]
        document["graph_entity_name"] = row["entity_name"]
        document["graph_entity_type"] = row["entity_type"]
        document["graph_mention_count"] = row["mention_count"]
        return document
