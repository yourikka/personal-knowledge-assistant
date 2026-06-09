from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import IngestRequest
from app.pipeline.orchestrator import KnowledgePipeline
from app.services.embedding_service import EmbeddingService
from app.services.query_cache import QueryCacheService
from app.services.vector_store import VectorStore


def build_pipeline(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        embedding_provider="local",
        embedding_api_key="",
        openai_api_key="",
        query_cache_enabled=True,
        query_cache_ttl_seconds=60,
        query_stream_chunk_chars=16,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, EmbeddingService(settings))
    return settings, KnowledgePipeline(settings, repo, vector_store)


def test_query_cache_service_returns_copied_cached_values(tmp_path):
    settings = Settings(sqlite_path=str(tmp_path / "knowledge.db"), query_cache_enabled=True)
    cache = QueryCacheService(settings)
    key = cache.make_key("LangGraph 怎么用", top_k=3, session_id="s1")

    cache.set(key, {"references": [{"id": "doc-1"}], "logs": []})
    cached = cache.get(key)
    cached["references"][0]["id"] = "changed"

    assert cache.get(key)["references"][0]["id"] == "doc-1"


def test_pipeline_query_uses_rag_cache_and_streams_events(tmp_path):
    _, pipeline = build_pipeline(tmp_path)
    pipeline.ingest(
        IngestRequest(
            source_type="text",
            source="技术：LangGraph\nLangGraph 适合构建多 Agent 工作流和 RAG 编排。",
            title="LangGraph 缓存测试",
        )
    )

    first = pipeline.query("LangGraph 适合做什么？", top_k=2)
    second = pipeline.query("LangGraph 适合做什么？", top_k=2)
    events = list(pipeline.query_stream("没有命中文档的问题", top_k=1))

    assert first["references"]
    assert any("cache: 命中高频 Query 缓存" in log for log in second["logs"])
    assert events[0]["event"] == "status"
    assert any(event["event"] == "delta" for event in events)
    assert events[-1]["event"] == "done"


def test_pipeline_query_fast_mode_skips_model_generation(tmp_path, monkeypatch):
    settings, pipeline = build_pipeline(tmp_path)
    pipeline.ingest(
        IngestRequest(
            source_type="text",
            source="LangGraph 可以把采集、解析、摘要和问答节点编排成多 Agent 工作流。",
            title="LangGraph 快速问答测试",
        )
    )
    settings.openai_api_key = "test-key"

    def fail_if_model_called(*_, **__):
        raise AssertionError("fast mode should not call model")

    monkeypatch.setattr(pipeline.openai_service, "generate_json", fail_if_model_called)
    monkeypatch.setattr(pipeline.openai_service, "generate_text", fail_if_model_called)

    result = pipeline.query("LangGraph 适合做什么？", top_k=2, answer_mode="fast")

    assert result["answer"].startswith("我在知识库里找到")
    assert result["references"]
    assert any("快速模式已使用本地 RAG 摘要" in log for log in result["logs"])


def test_pipeline_query_falls_back_when_model_generation_fails(tmp_path, monkeypatch):
    settings, pipeline = build_pipeline(tmp_path)
    pipeline.ingest(
        IngestRequest(
            source_type="text",
            source="LangGraph 可以把采集、解析、摘要和问答节点编排成多 Agent 工作流。",
            title="LangGraph 回退测试",
        )
    )

    settings.openai_api_key = "test-key"

    def fail_model_call(*_, **__):
        raise RuntimeError("model timeout")

    monkeypatch.setattr(pipeline.openai_service, "generate_json", fail_model_call)
    monkeypatch.setattr(pipeline.openai_service, "generate_text", fail_model_call)

    result = pipeline.query("LangGraph 适合做什么？", top_k=2, answer_mode="model")

    assert result["answer"].startswith("我在知识库里找到")
    assert result["references"]
    assert any("模型生成失败，已回退本地 RAG 摘要" in log for log in result["logs"])
