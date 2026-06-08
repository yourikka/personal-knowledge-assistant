from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.pipeline.orchestrator import KnowledgePipeline
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService
from app.services.openai_client import OpenAIService
from app.services.vector_store import VectorStore


def build_memory_service(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        openai_api_key="",
        memory_enabled=True,
        memory_top_k=3,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, embedding_service)
    service = MemoryService(settings, repo, vector_store, OpenAIService(settings))
    return settings, repo, vector_store, service


def test_memory_service_stores_and_retrieves_session_memory(tmp_path):
    _, repo, vector_store, service = build_memory_service(tmp_path)
    record = {
        "id": "mem-1",
        "session_id": "s1",
        "kind": "preference",
        "content": "用户偏好：回答要简洁，并使用中文。",
        "importance": 0.9,
        "tags": ["回答风格", "中文"],
        "metadata": {},
    }
    repo.upsert_memory(record)
    vector_store.add_memory(record["id"], record["content"], {"kind": "memory", "session_id": "s1"})

    memories = service.retrieve("以后回答风格怎么处理", session_id="s1", top_k=2)

    assert memories
    assert memories[0]["id"] == "mem-1"
    assert memories[0]["kind"] == "preference"


def test_memory_retrieval_respects_session_scope(tmp_path):
    _, repo, vector_store, service = build_memory_service(tmp_path)
    record = {
        "id": "mem-private",
        "session_id": "s1",
        "kind": "fact",
        "content": "这个会话的私有记忆只属于 s1。",
        "importance": 0.9,
        "tags": ["私有记忆"],
        "metadata": {},
    }
    repo.upsert_memory(record)
    vector_store.add_memory(record["id"], record["content"], {"kind": "memory", "session_id": "s1"})

    memories = service.retrieve("私有记忆", session_id="s2", top_k=2)

    assert memories == []


def test_vector_store_filters_memory_and_chunk_kinds(tmp_path):
    settings, _, vector_store, _ = build_memory_service(tmp_path)
    assert settings.memory_enabled is True
    vector_store.add_chunk("chunk-1", "LangGraph 文档切片", {"document_id": "doc-1"})
    vector_store.add_memory("mem-1", "LangGraph 用户偏好", {"session_id": "s1"})

    chunk_hits = vector_store.search("LangGraph", top_k=5, kind="chunk")
    memory_hits = vector_store.search("LangGraph", top_k=5, kind="memory")

    assert [item["id"] for item in chunk_hits] == ["chunk-1"]
    assert [item["id"] for item in memory_hits] == ["mem-1"]


def test_query_pipeline_writes_rule_based_memory(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        openai_api_key="",
        memory_enabled=True,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, embedding_service)
    pipeline = KnowledgePipeline(settings, repo, vector_store)

    result = pipeline.query("记住：我希望默认用中文简洁回答。", top_k=3, session_id="web-session")

    memories = repo.list_memories(session_id="web-session")
    assert result["memories"] == []
    assert len(memories) == 1
    assert memories[0]["scope"] == "user"
    assert memories[0]["session_id"] is None
    assert "中文简洁回答" in memories[0]["content"]


def test_memory_records_can_be_listed_and_deleted(tmp_path):
    _, repo, _, _ = build_memory_service(tmp_path)
    repo.upsert_memory(
        {
            "id": "mem-delete",
            "session_id": "web-session",
            "scope": "session",
            "kind": "fact",
            "content": "这条记忆用于前端管理删除测试。",
            "importance": 0.7,
            "tags": ["管理"],
            "metadata": {},
        }
    )

    memories = repo.list_memories(session_id="web-session")

    assert [memory["id"] for memory in memories] == ["mem-delete"]
    assert repo.delete_memory("mem-delete") is True
    assert repo.list_memories(session_id="web-session") == []
    assert repo.delete_memory("mem-delete") is False


def test_memory_records_can_be_updated(tmp_path):
    _, repo, _, _ = build_memory_service(tmp_path)
    repo.upsert_memory(
        {
            "id": "mem-edit",
            "session_id": "web-session",
            "scope": "session",
            "kind": "preference",
            "content": "用户希望回答非常详细。",
            "importance": 0.4,
            "tags": ["旧标签"],
            "metadata": {},
        }
    )

    updated = repo.update_memory(
        "mem-edit",
        content="用户希望回答简洁，并优先使用中文。",
        importance=0.88,
        tags=["中文", "简洁"],
    )

    assert updated is not None
    assert updated["content"] == "用户希望回答简洁，并优先使用中文。"
    assert updated["importance"] == 0.88
    assert updated["tags"] == ["中文", "简洁"]
    assert updated["kind"] == "preference"
    assert updated["session_id"] == "web-session"
    assert repo.update_memory("missing-memory", content="不存在") is None


def test_memory_service_supersedes_conflicting_memories(tmp_path):
    _, repo, _, service = build_memory_service(tmp_path)

    first = service.learn_from_turn(
        query="记住：我希望默认用中文简洁回答。",
        answer="已记录。",
        session_id="s1",
        references=[],
    )
    second = service.learn_from_turn(
        query="记住：我希望默认用英文简洁回答。",
        answer="已记录。",
        session_id="s1",
        references=[],
    )

    all_memories = repo.list_memories(limit=10, include_global=True)
    assert first
    assert second
    assert len(all_memories) == 1
    assert all_memories[0]["status"] == "active"
    assert "英文简洁回答" in all_memories[0]["content"]


def test_query_pipeline_can_answer_from_memory_without_documents(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        openai_api_key="",
        memory_enabled=True,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, embedding_service)
    pipeline = KnowledgePipeline(settings, repo, vector_store)
    memory = {
        "id": "mem-answer",
        "session_id": "web-session",
        "kind": "preference",
        "content": "用户希望默认使用中文简洁回答。",
        "importance": 0.9,
        "tags": ["中文", "简洁"],
        "metadata": {},
    }
    repo.upsert_memory(memory)
    vector_store.add_memory(memory["id"], memory["content"], {"kind": "memory", "session_id": "web-session"})

    result = pipeline.query("我希望你默认怎么回答？", top_k=3, session_id="web-session")

    assert "当前没有找到相关文档" in result["answer"]
    assert "中文简洁回答" in result["answer"]
    assert result["references"] == []
    assert result["memories"][0]["id"] == "mem-answer"
