from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import IngestRequest
from app.pipeline.orchestrator import KnowledgePipeline
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def build_pipeline(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        embedding_provider="local",
        embedding_api_key="",
        openai_api_key="",
        related_score_threshold=0.01,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, EmbeddingService(settings))
    pipeline = KnowledgePipeline(settings, repo, vector_store)
    return repo, vector_store, pipeline


def test_reindex_document_updates_only_target_document_indexes(tmp_path):
    repo, vector_store, pipeline = build_pipeline(tmp_path)
    first = pipeline.ingest(
        IngestRequest(source_type="text", source="# LangGraph\n技术：LangGraph\n旧内容。", title="LangGraph")
    )
    second = pipeline.ingest(
        IngestRequest(source_type="text", source="# Chroma\n技术：Chroma\n向量检索内容。", title="Chroma")
    )
    document = repo.get_document(first["document_id"])
    repo.upsert_document(
        {
            **document,
            "raw_text": "# LangGraph\n技术：LangGraph\n新增增量索引内容。",
            "cleaned_text": "# LangGraph\n技术：LangGraph\n新增增量索引内容。",
        }
    )

    result = pipeline.reindex_document(first["document_id"])
    chunks = repo.list_document_chunks(first["document_id"])

    assert result["status"] == "ok"
    assert result["document_id"] == first["document_id"]
    assert result["chunks"] == len(chunks)
    assert "新增增量索引内容" in chunks[0]["text"]
    assert second["document_id"] in vector_store.local_embeddings
    assert first["document_id"] in vector_store.local_embeddings


def test_replace_links_removes_stale_reverse_edges(tmp_path):
    repo, _, _ = build_pipeline(tmp_path)
    for doc_id in ("doc-a", "doc-b", "doc-c"):
        repo.upsert_document(
            {
                "id": doc_id,
                "fingerprint": f"fp-{doc_id}",
                "source_type": "text",
                "source_uri": f"inline://{doc_id}",
                "title": doc_id,
                "raw_text": doc_id,
                "cleaned_text": doc_id,
                "summary": doc_id,
                "category": "技术",
                "confidence": 0.9,
                "tags": [doc_id],
                "metadata": {},
            }
        )

    repo.replace_links("doc-a", [{"target_id": "doc-b", "score": 0.8}])
    repo.replace_links("doc-a", [{"target_id": "doc-c", "score": 0.7}])

    assert repo.list_links("doc-b") == []
    assert [link["target_id"] for link in repo.list_links("doc-a")] == ["doc-c"]


def test_rebuild_links_refreshes_all_document_links(tmp_path):
    repo, _, pipeline = build_pipeline(tmp_path)
    first = pipeline.ingest(
        IngestRequest(source_type="text", source="LangGraph 用于多 Agent 编排。", title="LangGraph A")
    )
    second = pipeline.ingest(
        IngestRequest(source_type="text", source="LangGraph 支持流程节点编排。", title="LangGraph B")
    )
    repo.replace_links(first["document_id"], [])

    result = pipeline.rebuild_links()

    assert result["status"] == "ok"
    assert result["documents"] == 2
    assert result["links_rebuilt"] > 0
    assert repo.list_links(first["document_id"]) or repo.list_links(second["document_id"])
