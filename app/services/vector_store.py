from __future__ import annotations

import os
from typing import Any

from .embedding_service import EmbeddingService
from .text_utils import cosine_similarity, overlap_score

try:
    import chromadb
    from chromadb.errors import InvalidDimensionException
except ImportError:
    chromadb = None
    InvalidDimensionException = None


class VectorStore:
    def __init__(self, chroma_dir: str, enable_chroma: bool, embedding_service: EmbeddingService) -> None:
        self.enable_chroma = bool(enable_chroma and chromadb is not None)
        self.embedding_service = embedding_service
        self.local_embeddings: dict[str, list[float]] = {}
        self.local_texts: dict[str, str] = {}
        self.local_metadata: dict[str, dict[str, Any]] = {}
        self._text_embedding_cache: dict[str, list[float]] = {}
        self.client = None
        self.collection = None
        self.collection_name = "knowledge_documents"

        if self.enable_chroma:
            os.makedirs(chroma_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=chroma_dir)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self._ensure_collection_dimension()

    def add_text(self, item_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.add_texts([(item_id, text, metadata or {})])

    def add_texts(self, items: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        if not items:
            return
        ids = [item_id for item_id, _, _ in items]
        documents = [text for _, text, _ in items]
        metadatas = [metadata or {} for _, _, metadata in items]
        embeddings = [self._embed_text(text) for text in documents]

        for item_id, text, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.local_embeddings[item_id] = embedding
            self.local_texts[item_id] = text
            self.local_metadata[item_id] = metadata

        if self.collection is not None:
            self._upsert_collection(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def add_document(self, document_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {"kind": "document", **(metadata or {})}
        self.add_text(document_id, text, payload)

    def add_chunk(self, chunk_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {"kind": "chunk", **(metadata or {})}
        self.add_text(chunk_id, text, payload)

    def add_chunks(self, chunks: list[dict[str, Any]], metadata_factory) -> None:
        self.add_texts(
            [
                (chunk["id"], chunk["text"], {"kind": "chunk", **metadata_factory(chunk)})
                for chunk in chunks
            ]
        )

    def add_section(self, section_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {"kind": "section", **(metadata or {})}
        self.add_text(section_id, text, payload)

    def add_sections(self, sections: list[dict[str, Any]], metadata_factory) -> None:
        self.add_texts(
            [
                (section["id"], section["text"], {"kind": "section", **metadata_factory(section)})
                for section in sections
            ]
        )

    def add_memory(self, memory_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {"kind": "memory", **(metadata or {})}
        self.add_text(memory_id, text, payload)

    def reset(self) -> None:
        self.local_embeddings = {}
        self.local_texts = {}
        self.local_metadata = {}
        self._text_embedding_cache = {}

    def delete_ids(self, ids: list[str]) -> None:
        for item_id in ids:
            self.local_embeddings.pop(item_id, None)
            self.local_texts.pop(item_id, None)
            self.local_metadata.pop(item_id, None)

        if self.collection is not None and ids:
            self.collection.delete(ids=ids)

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for metadata in self.local_metadata.values():
            kind = str(metadata.get("kind") or "unknown")
            by_kind[kind] = by_kind.get(kind, 0) + 1
        chroma_items = self.collection.count() if self.collection is not None else 0
        return {
            "local_items": len(self.local_embeddings),
            "chroma_items": int(chroma_items),
            "by_kind": by_kind,
        }

    def search(
        self,
        query: str,
        top_k: int,
        exclude_ids: set[str] | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        exclude_ids = exclude_ids or set()
        query_embedding = self._embed_text(query)
        local_ranked = []
        for document_id, document_embedding in self.local_embeddings.items():
            if document_id in exclude_ids:
                continue
            metadata = self.local_metadata.get(document_id, {})
            if kind and metadata.get("kind") != kind:
                continue
            vector_score = cosine_similarity(query_embedding, document_embedding)
            lexical_score = overlap_score(query, self.local_texts.get(document_id, ""))
            score = vector_score * 0.7 + lexical_score * 0.3
            if score > 0:
                local_ranked.append({"id": document_id, "score": round(score, 4)})
        local_ranked.sort(key=lambda item: item["score"], reverse=True)

        if self.collection is None:
            return local_ranked[:top_k]

        n_results = min(max(top_k * 2, 6), max(1, len(self.local_embeddings)))
        chroma_ranked: dict[str, float] = {item["id"]: item["score"] for item in local_ranked}
        where = {"kind": kind} if kind else None
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
        except InvalidDimensionException:
            self._recreate_collection()
            self._sync_collection()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for document_id, distance in zip(ids, distances):
            if document_id in exclude_ids:
                continue
            score = 1.0 / (1.0 + float(distance))
            chroma_ranked[document_id] = max(chroma_ranked.get(document_id, 0.0), round(score, 4))

        merged = [{"id": doc_id, "score": score} for doc_id, score in chroma_ranked.items() if score > 0]
        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[:top_k]

    def similarity(self, left_text: str, right_text: str) -> float:
        vector_score = cosine_similarity(self._embed_text(left_text), self._embed_text(right_text))
        lexical_score = overlap_score(left_text, right_text)
        return round(vector_score * 0.7 + lexical_score * 0.3, 4)

    def _expected_embedding_dimension(self) -> int:
        return len(self._embed_text(""))

    def _embed_text(self, text: str) -> list[float]:
        key = text.strip()
        if key in self._text_embedding_cache:
            return self._text_embedding_cache[key]
        embedding = self.embedding_service.embed(text)
        if key:
            self._text_embedding_cache[key] = embedding
        return embedding

    def _collection_dimension(self) -> int | None:
        if self.collection is None:
            return None
        model = getattr(self.collection, "_model", None)
        return getattr(model, "dimension", None)

    def _ensure_collection_dimension(self) -> None:
        actual = self._collection_dimension()
        expected = self._expected_embedding_dimension()
        if actual is not None and actual != expected:
            # Chroma collection dimension is fixed after the first insert. If the
            # embedding configuration changes, rebuild the persisted collection and
            # let pipeline bootstrap repopulate it from SQLite.
            self._recreate_collection()

    def _recreate_collection(self) -> None:
        if self.client is None:
            return
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def _sync_collection(self) -> None:
        if self.collection is None or not self.local_embeddings:
            return
        self.collection.upsert(
            ids=list(self.local_embeddings.keys()),
            documents=[self.local_texts[item_id] for item_id in self.local_embeddings],
            metadatas=[self.local_metadata[item_id] for item_id in self.local_embeddings],
            embeddings=[self.local_embeddings[item_id] for item_id in self.local_embeddings],
        )

    def _upsert_collection(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except InvalidDimensionException:
            self._recreate_collection()
            self._sync_collection()
