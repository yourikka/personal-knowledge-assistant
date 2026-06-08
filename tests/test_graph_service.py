from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import IngestRequest
from app.pipeline.orchestrator import KnowledgePipeline
from app.services.embedding_service import EmbeddingService
from app.services.graph_service import GraphExtractionService
from app.services.vector_store import VectorStore


def build_pipeline(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        openai_api_key="",
        graph_enabled=True,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, EmbeddingService(settings))
    pipeline = KnowledgePipeline(settings, repo, vector_store)
    return settings, repo, pipeline


def test_ingest_extracts_document_graph(tmp_path):
    _, repo, pipeline = build_pipeline(tmp_path)

    result = pipeline.ingest(
        IngestRequest(
            source_type="text",
            source=(
                "技术：LangGraph\n"
                "组织：OpenAI\n"
                "概念：分层检索\n"
                "LangGraph 可以编排多个 Agent，OpenAI 模型用于摘要和问答。"
            ),
            title="LangGraph 知识库实践",
        )
    )

    entities = repo.list_document_entities(result["document_id"])
    edges = repo.list_document_graph_edges(result["document_id"])
    graph = pipeline.graph_service.graph_view(result["document_id"])

    assert any(entity["name"] == "LangGraph" and entity["entity_type"] == "technology" for entity in entities)
    assert any(entity["name"] == "OpenAI" and entity["entity_type"] == "organization" for entity in entities)
    assert edges
    assert any(node["name"] == "LangGraph" for node in graph["nodes"])
    assert any(edge["relation"] for edge in graph["edges"])
    assert any(node["id"].startswith("ent-") for node in result["graph"]["nodes"])


def test_graph_retrieval_returns_entity_documents(tmp_path):
    settings, repo, pipeline = build_pipeline(tmp_path)
    result = pipeline.ingest(
        IngestRequest(
            source_type="text",
            source="技术：LangGraph\n概念：Agent 编排\nLangGraph 用于构建 StateGraph 多 Agent 流水线。",
            title="Agent 编排笔记",
        )
    )
    graph_service = GraphExtractionService(settings, repo)

    documents = graph_service.related_documents("LangGraph 怎么做 Agent 编排？", limit=5)

    assert [document["id"] for document in documents] == [result["document_id"]]
    assert documents[0]["graph_entity_name"] == "LangGraph"
