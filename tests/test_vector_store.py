from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.services.embedding_service import EmbeddingService
import app.services.vector_store as vector_store_module
from app.services.vector_store import VectorStore


class CountingEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0] if "LangGraph" in text else [0.0, 1.0]


def test_vector_store_recovers_from_persisted_chroma_dimension_mismatch(tmp_path):
    chroma_dir = tmp_path / "chroma"
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
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


def test_vector_store_reuses_text_embeddings(tmp_path):
    settings = Settings(chroma_dir=str(tmp_path / "chroma"), enable_chroma=False)
    embedding_service = CountingEmbeddingService()
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, embedding_service)

    vector_store.similarity("LangGraph", "LangGraph")
    vector_store.similarity("LangGraph", "LangGraph")

    assert embedding_service.calls == ["LangGraph"]


def test_vector_store_batches_collection_upserts(tmp_path):
    class FakeCollection:
        def __init__(self) -> None:
            self.calls = []

        def upsert(self, ids, documents, metadatas, embeddings):
            self.calls.append({"ids": ids, "documents": documents, "metadatas": metadatas, "embeddings": embeddings})

    settings = Settings(chroma_dir=str(tmp_path / "chroma"), enable_chroma=False)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, CountingEmbeddingService())
    collection = FakeCollection()
    vector_store.collection = collection

    vector_store.add_texts(
        [
            ("doc-1", "LangGraph 文档", {"kind": "document"}),
            ("doc-2", "Chroma 文档", {"kind": "document"}),
        ]
    )

    assert len(collection.calls) == 1
    assert collection.calls[0]["ids"] == ["doc-1", "doc-2"]
    assert set(vector_store.local_embeddings) == {"doc-1", "doc-2"}


def test_vector_store_disables_chroma_telemetry(tmp_path, monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, path, settings):
            captured["path"] = path
            captured["settings"] = settings

        def get_or_create_collection(self, name):
            captured["collection_name"] = name
            return FakeCollection()

    class FakeCollection:
        _model = None

    class FakeChroma:
        PersistentClient = FakeClient

    settings = Settings(chroma_dir=str(tmp_path / "chroma"), enable_chroma=True)
    monkeypatch.setattr(vector_store_module, "chromadb", FakeChroma)

    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, CountingEmbeddingService())

    assert vector_store.enable_chroma is True
    assert captured["path"] == settings.chroma_dir
    assert captured["collection_name"] == "knowledge_documents"
    assert captured["settings"].anonymized_telemetry is False
