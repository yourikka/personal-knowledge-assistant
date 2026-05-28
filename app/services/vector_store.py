from __future__ import annotations

import os
from typing import Any

from .embedding_service import EmbeddingService
from .text_utils import cosine_similarity, overlap_score

try:
    import chromadb
except ImportError:
    chromadb = None


class VectorStore:
    def __init__(self, chroma_dir: str, enable_chroma: bool, embedding_service: EmbeddingService) -> None:
        self.enable_chroma = bool(enable_chroma and chromadb is not None)
        self.embedding_service = embedding_service
        self.local_embeddings: dict[str, list[float]] = {}
        self.local_texts: dict[str, str] = {}
        self.local_metadata: dict[str, dict[str, Any]] = {}
        self.collection = None

        if self.enable_chroma:
            os.makedirs(chroma_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=chroma_dir)
            self.collection = client.get_or_create_collection(name="knowledge_documents")

    def add_text(self, item_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        embedding = self.embedding_service.embed(text)
        self.local_embeddings[item_id] = embedding
        self.local_texts[item_id] = text
        self.local_metadata[item_id] = metadata or {}

        if self.collection is not None:
            self.collection.upsert(
                ids=[item_id],
                documents=[text],
                metadatas=[metadata or {}],
                embeddings=[embedding],
            )

    def add_document(self, document_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {"kind": "document", **(metadata or {})}
        self.add_text(document_id, text, payload)

    def add_chunk(self, chunk_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {"kind": "chunk", **(metadata or {})}
        self.add_text(chunk_id, text, payload)

    def add_section(self, section_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {"kind": "section", **(metadata or {})}
        self.add_text(section_id, text, payload)

    def add_memory(self, memory_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        payload = {"kind": "memory", **(metadata or {})}
        self.add_text(memory_id, text, payload)

    def reset(self) -> None:
        self.local_embeddings = {}
        self.local_texts = {}
        self.local_metadata = {}

    def delete_ids(self, ids: list[str]) -> None:
        for item_id in ids:
            self.local_embeddings.pop(item_id, None)
            self.local_texts.pop(item_id, None)
            self.local_metadata.pop(item_id, None)

        if self.collection is not None and ids:
            self.collection.delete(ids=ids)

    def search(
        self,
        query: str,
        top_k: int,
        exclude_ids: set[str] | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        exclude_ids = exclude_ids or set()
        query_embedding = self.embedding_service.embed(query)
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
        vector_score = cosine_similarity(self.embedding_service.embed(left_text), self.embedding_service.embed(right_text))
        lexical_score = overlap_score(left_text, right_text)
        return round(vector_score * 0.7 + lexical_score * 0.3, 4)
