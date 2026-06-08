from __future__ import annotations

import chromadb

from app.config import Settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def test_vector_store_recovers_from_persisted_chroma_dimension_mismatch(tmp_path):
    chroma_dir = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    legacy = client.get_or_create_collection(name="knowledge_documents")
    legacy.upsert(
        ids=["legacy-doc"],
        documents=["legacy text"],
        metadatas=[{"kind": "document"}],
        embeddings=[[0.0] * 64],
    )

    settings = Settings(
        chroma_dir=str(chroma_dir),
        enable_chroma=True,
        embedding_provider="local",
        embedding_api_key="",
        openai_api_key="",
    )
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, EmbeddingService(settings))

    vector_store.add_document("doc-1", "LangGraph 适合多 Agent 编排", {"title": "LangGraph"})
    results = vector_store.search("LangGraph", top_k=1, kind="document")

    assert vector_store.collection is not None
    assert vector_store.collection.count() == 1
    persisted = vector_store.client.get_collection("knowledge_documents")
    assert getattr(persisted._model, "dimension", None) == 128
    assert results[0]["id"] == "doc-1"
