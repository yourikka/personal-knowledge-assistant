from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import IngestRequest
from app.pipeline.orchestrator import KnowledgePipeline
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def test_repository_and_vector_store_stats_reflect_indexed_content(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        openai_api_key="",
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, EmbeddingService(settings))
    pipeline = KnowledgePipeline(settings, repo, vector_store)

    pipeline.ingest(
        IngestRequest(
            source_type="markdown",
            source="# LangGraph\n\nLangGraph 适合做多 Agent 编排。",
            title="LangGraph 状态测试",
        )
    )

    repository_stats = repo.stats()
    vector_stats = vector_store.stats()

    assert repository_stats["documents"] == 1
    assert repository_stats["chunks"] >= 1
    assert repository_stats["sections"] >= 1
    assert vector_stats["local_items"] >= 3
    assert vector_stats["by_kind"]["document"] == 1
    assert vector_stats["by_kind"]["chunk"] >= 1
