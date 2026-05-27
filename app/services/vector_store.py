from __future__ import annotations

import os
from typing import Any

from .text_utils import cosine_similarity, make_hash_embedding, overlap_score

try:
    import chromadb
except ImportError:
    chromadb = None


class VectorStore:
    def __init__(self, chroma_dir: str, enable_chroma: bool) -> None:
        self.enable_chroma = bool(enable_chroma and chromadb is not None)
        self.local_embeddings: dict[str, list[float]] = {}
        self.local_texts: dict[str, str] = {}
        self.collection = None

        if self.enable_chroma:
            os.makedirs(chroma_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=chroma_dir)
            self.collection = client.get_or_create_collection(name="knowledge_documents")

    def add_document(self, document_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        embedding = make_hash_embedding(text)
        self.local_embeddings[document_id] = embedding
        self.local_texts[document_id] = text

        if self.collection is not None:
            self.collection.upsert(
                ids=[document_id],
                documents=[text],
                metadatas=[metadata or {}],
                embeddings=[embedding],
            )

    def reset(self) -> None:
        self.local_embeddings = {}
        self.local_texts = {}

    def search(self, query: str, top_k: int, exclude_ids: set[str] | None = None) -> list[dict[str, Any]]:
        exclude_ids = exclude_ids or set()
        query_embedding = make_hash_embedding(query)
        local_ranked = []
        for document_id, document_embedding in self.local_embeddings.items():
            if document_id in exclude_ids:
                continue
            vector_score = cosine_similarity(query_embedding, document_embedding)
            lexical_score = overlap_score(query, self.local_texts.get(document_id, ""))
            score = vector_score * 0.7 + lexical_score * 0.3
            if score > 0:
                local_ranked.append({"id": document_id, "score": round(score, 4)})
        local_ranked.sort(key=lambda item: item["score"], reverse=True)

        if self.collection is None:
            return local_ranked[:top_k]

        chroma_ranked: dict[str, float] = {item["id"]: item["score"] for item in local_ranked}
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * 2, 6),
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
