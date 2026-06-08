from __future__ import annotations

from app.config import Settings
from app.services.embedding_service import EmbeddingService


def test_embedding_service_falls_back_to_local_hash_embedding():
    settings = Settings(
        embedding_provider="local",
        embedding_api_key="",
    )
    service = EmbeddingService(settings)

    vector = service.embed("LangGraph RAG chunk retrieval")

    assert isinstance(vector, list)
    assert len(vector) == 128
    assert any(value != 0 for value in vector)


def test_embedding_service_returns_configured_zero_vector_for_blank_input():
    settings = Settings(
        embedding_provider="zhipu",
        embedding_api_key="dummy",
        embedding_dimensions=2048,
    )
    service = EmbeddingService(settings)

    vector = service.embed("   ")

    assert len(vector) == 2048
    assert set(vector) == {0.0}


def test_embedding_service_falls_back_when_remote_provider_fails(monkeypatch):
    settings = Settings(
        embedding_provider="zhipu",
        embedding_api_key="dummy",
        embedding_dimensions=16,
    )
    service = EmbeddingService(settings)

    def fail_request(*_, **__):
        raise RuntimeError("network down")

    monkeypatch.setattr(service, "_post_json", fail_request)

    vector = service.embed("LangGraph RAG")

    assert len(vector) == 16
    assert any(value != 0 for value in vector)
