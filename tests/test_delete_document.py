from __future__ import annotations

from app.db import KnowledgeRepository
from app.pipeline.orchestrator import KnowledgePipeline
from app.config import Settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def build_pipeline(tmp_path):
    sqlite_path = tmp_path / "knowledge.db"
    chroma_dir = tmp_path / "chroma"
    settings = Settings(
        sqlite_path=str(sqlite_path),
        chroma_dir=str(chroma_dir),
        enable_chroma=False,
        openai_api_key="",
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, embedding_service)
    pipeline = KnowledgePipeline(settings, repo, vector_store)
    return repo, vector_store, pipeline


def seed_document(repo: KnowledgeRepository, vector_store: VectorStore, document_id: str = "doc-1"):
    repo.upsert_document(
        {
            "id": document_id,
            "fingerprint": f"fp-{document_id}",
            "source_type": "markdown",
            "source_uri": f"inline://{document_id}",
            "title": "测试文档",
            "raw_text": "原始文本",
            "cleaned_text": "清洗后的测试文档内容",
            "summary": "摘要",
            "category": "技术",
            "confidence": 0.92,
            "tags": ["测试", "删除"],
            "metadata": {},
        }
    )
    chunks = [
        {
            "id": f"{document_id}:chunk:0000",
            "chunk_index": 0,
            "text": "第一段切片",
            "char_start": 0,
            "char_end": 5,
            "metadata": {"heading_path": ["标题"]},
        },
        {
            "id": f"{document_id}:chunk:0001",
            "chunk_index": 1,
            "text": "第二段切片",
            "char_start": 6,
            "char_end": 11,
            "metadata": {"heading_path": ["标题"]},
        },
    ]
    repo.replace_document_chunks(document_id, chunks)
    sections = [
        {
            "id": f"{document_id}:section:0000",
            "document_id": document_id,
            "section_index": 0,
            "heading": "标题",
            "heading_path": ["标题"],
            "text": "第一段切片 第二段切片",
            "char_start": 0,
            "char_end": 11,
            "metadata": {"chunk_ids": [chunk["id"] for chunk in chunks]},
        }
    ]
    repo.replace_document_sections(document_id, sections)
    repo.replace_links(document_id, [{"target_id": document_id, "score": 0.9}])

    vector_store.add_document(document_id, "清洗后的测试文档内容", {"title": "测试文档"})
    for section in sections:
        vector_store.add_section(section["id"], section["text"], {"document_id": document_id})
    for chunk in chunks:
        vector_store.add_chunk(chunk["id"], chunk["text"], {"document_id": document_id})

    return chunks, sections


def test_delete_document_removes_repo_rows_and_vector_entries(tmp_path):
    repo, vector_store, pipeline = build_pipeline(tmp_path)
    chunks, sections = seed_document(repo, vector_store)

    result = pipeline.delete_document("doc-1")

    assert result["status"] == "ok"
    assert result["document_id"] == "doc-1"
    assert result["deleted_chunk_ids"] == [chunk["id"] for chunk in chunks]
    assert repo.get_document("doc-1") is None
    assert repo.list_document_chunks("doc-1") == []
    assert repo.list_document_sections("doc-1") == []
    assert repo.list_links("doc-1") == []
    assert "doc-1" not in vector_store.local_embeddings
    for section in sections:
        assert section["id"] not in vector_store.local_embeddings
    for chunk in chunks:
        assert chunk["id"] not in vector_store.local_embeddings


def test_delete_document_raises_for_missing_document(tmp_path):
    _, _, pipeline = build_pipeline(tmp_path)

    try:
        pipeline.delete_document("missing-doc")
    except ValueError as error:
        assert str(error) == "文档不存在。"
    else:
        raise AssertionError("expected ValueError for missing document")
