from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository, utc_now
from app.services.embedding_service import EmbeddingService
from app.services.openai_client import OpenAIService
from app.services.personalization_service import PersonalizationService
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore


def build_services(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        openai_api_key="",
        personalization_boost=0.5,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, EmbeddingService(settings))
    personalization = PersonalizationService(settings, repo)
    rag = RAGService(settings, repo, vector_store, OpenAIService(settings), personalization_service=personalization)
    return repo, personalization, rag


def test_personalization_records_profile_clicks_and_feedback(tmp_path):
    repo, personalization, _ = build_services(tmp_path)
    document = make_document("doc-langgraph", "LangGraph", ["langgraph"])
    repo.upsert_document(document)

    personalization.learn_query("s1", "LangGraph Agent 编排")
    personalization.record_click("s1", "doc-langgraph", "Agent 编排")
    personalization.record_feedback("s1", "Agent 编排", rating=1, document_id="doc-langgraph", comment="有用")

    profile = repo.list_query_profile("s1")
    clicks = repo.document_click_counts("s1")

    assert profile
    assert clicks == {"doc-langgraph": 1}
    assert personalization.score_document("s1", document) > 0


def test_rag_rerank_uses_session_personalization(tmp_path):
    repo, _, rag = build_services(tmp_path)
    langgraph_doc = make_document("doc-langgraph", "LangGraph 实践", ["langgraph"])
    chroma_doc = make_document("doc-chroma", "Chroma 实践", ["chroma"])
    repo.record_query_profile("s-langgraph", ["langgraph"], weight=5)
    repo.record_query_profile("s-chroma", ["chroma"], weight=5)

    candidates = {
        "chunk-langgraph": make_candidate("chunk-langgraph", langgraph_doc),
        "chunk-chroma": make_candidate("chunk-chroma", chroma_doc),
    }

    langgraph_ranked = rag._rerank("怎么做知识库", candidates, session_id="s-langgraph")
    chroma_ranked = rag._rerank("怎么做知识库", candidates, session_id="s-chroma")

    assert langgraph_ranked[0]["document"]["id"] == "doc-langgraph"
    assert chroma_ranked[0]["document"]["id"] == "doc-chroma"


def make_document(document_id: str, title: str, tags: list[str]) -> dict:
    now = utc_now()
    return {
        "id": document_id,
        "fingerprint": f"fp-{document_id}",
        "source_type": "text",
        "source_uri": f"inline://{document_id}",
        "title": title,
        "raw_text": title,
        "cleaned_text": title,
        "summary": title,
        "category": "技术",
        "confidence": 0.9,
        "tags": tags,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }


def make_candidate(chunk_id: str, document: dict) -> dict:
    return {
        "chunk": {
            "id": chunk_id,
            "document_id": document["id"],
            "chunk_index": 0,
            "text": "知识库 实践",
            "char_start": 0,
            "char_end": 5,
            "metadata": {},
            "created_at": document["created_at"],
        },
        "document": document,
        "signals": [{"source": "chunk_keyword", "query": "知识库", "score": 0.5, "rank": 1}],
        "best_rank": 1,
        "raw_score": 0.5,
    }
