from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.services.chunking import DocumentChunker
from app.services.embedding_service import EmbeddingService
from app.services.openai_client import OpenAIService
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore


def build_rag(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        openai_api_key="",
        rag_hierarchical_enabled=True,
        rag_document_top_k=2,
        rag_section_top_k=3,
        rag_section_chunk_limit=2,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, embedding_service)
    rag_service = RAGService(settings, repo, vector_store, OpenAIService(settings))
    return repo, vector_store, rag_service


def test_chunker_builds_sections_from_heading_paths():
    chunker = DocumentChunker(target_chars=120, overlap_chars=0, min_chars=40, max_chars=180)
    text = "# RAG\n\n检索增强生成需要切片。\n\n## 重排\n\n重排用于提升引用质量。\n\n## 记忆\n\n记忆用于保存偏好。"

    sections = chunker.sections("doc-1", text)

    assert len(sections) >= 2
    assert sections[0]["id"].startswith("doc-1:section:")
    assert sections[0]["metadata"]["unit_count"] >= 1
    assert all(section["char_start"] <= section["char_end"] for section in sections)


def test_rag_can_retrieve_chunk_from_section_hit(tmp_path):
    repo, vector_store, rag_service = build_rag(tmp_path)
    repo.upsert_document(
        {
            "id": "doc-auth",
            "fingerprint": "fp-doc-auth",
            "source_type": "markdown",
            "source_uri": "inline://doc-auth",
            "title": "认证笔记",
            "raw_text": "token handling details",
            "cleaned_text": "token handling details",
            "summary": "认证相关笔记",
            "category": "技术",
            "confidence": 0.9,
            "tags": ["认证"],
            "metadata": {},
        }
    )
    chunk = {
        "id": "doc-auth:chunk:0000",
        "document_id": "doc-auth",
        "chunk_index": 0,
        "text": "token handling details",
        "char_start": 0,
        "char_end": 22,
        "metadata": {"heading_path": ["认证"]},
    }
    section = {
        "id": "doc-auth:section:0000",
        "document_id": "doc-auth",
        "section_index": 0,
        "heading": "OAuth",
        "heading_path": ["OAuth"],
        "text": "OAuth token handling details",
        "char_start": 0,
        "char_end": 22,
        "metadata": {"chunk_ids": [chunk["id"]]},
    }
    repo.replace_document_chunks("doc-auth", [chunk])
    repo.replace_document_sections("doc-auth", [section])
    vector_store.add_section(section["id"], section["text"], {"document_id": "doc-auth", "section_index": 0})

    result = rag_service.retrieve("OAuth", top_k=1)

    assert result["references"]
    assert result["references"][0]["chunk_id"] == chunk["id"]
    signal_sources = {signal["source"] for signal in result["references"][0]["signals"]}
    assert "section_vector" in signal_sources
