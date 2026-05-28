from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import PipelineState
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore


class LinkingAgent:
    def __init__(
        self,
        settings: Settings,
        repo: KnowledgeRepository,
        vector_store: VectorStore,
        rag_service: RAGService | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.vector_store = vector_store
        self.rag_service = rag_service

    def run(self, state: PipelineState) -> PipelineState:
        self.vector_store.add_document(
            document_id=state.document_id,
            text=state.cleaned_text,
            metadata={"title": state.title, "category": state.category},
        )
        for section in state.sections:
            self.vector_store.add_section(
                section_id=section["id"],
                text=section["text"],
                metadata={
                    "document_id": state.document_id,
                    "section_index": section["section_index"],
                    "heading": section["heading"],
                    "title": state.title,
                    "category": state.category,
                },
            )
        for chunk in state.chunks:
            self.vector_store.add_chunk(
                chunk_id=chunk["id"],
                text=chunk["text"],
                metadata={
                    "document_id": state.document_id,
                    "chunk_index": chunk["chunk_index"],
                    "title": state.title,
                    "category": state.category,
                },
            )

        if self.rag_service:
            retrieval = self.rag_service.retrieve(
                query=f"{state.title}\n{state.summary}\n{' '.join(state.tags)}\n{state.cleaned_text[:2000]}",
                top_k=self.settings.related_top_k,
                exclude_ids={state.document_id},
            )
            grouped: dict[str, dict] = {}
            for item in retrieval["references"]:
                current = grouped.get(item["id"])
                if current is None or item["score"] > current["score"]:
                    grouped[item["id"]] = {
                        "id": item["id"],
                        "score": item["score"],
                        "signals": item.get("signals", []),
                        "chunk_id": item.get("chunk_id"),
                    }
            results = sorted(grouped.values(), key=lambda item: item["score"], reverse=True)
        else:
            results = self.vector_store.search(
                query=state.cleaned_text,
                top_k=self.settings.related_top_k + 1,
                exclude_ids={state.document_id},
            )
        related = []
        for item in results:
            if item["score"] < self.settings.related_score_threshold:
                continue
            document = self.repo.get_document(item["id"])
            if not document:
                continue
            related.append(
                {
                    "target_id": document["id"],
                    "source_id": state.document_id,
                    "title": document["title"],
                    "score": item["score"],
                    "summary": document["summary"],
                    "source_uri": document["source_uri"],
                    "signals": item.get("signals", []),
                }
            )

        state.related = related
        state.graph = {
            "nodes": [
                {"id": state.document_id, "title": state.title, "category": state.category},
                *[
                    {
                        "id": item["target_id"],
                        "title": item["title"],
                        "category": self.repo.get_document(item["target_id"])["category"],
                    }
                    for item in related
                    if self.repo.get_document(item["target_id"])
                ],
            ],
            "edges": [
                {
                    "source": state.document_id,
                    "target": item["target_id"],
                    "score": item["score"],
                    "signals": item.get("signals", []),
                    "type": "similar_to",
                }
                for item in related
            ],
        }
        self.repo.replace_links(
            source_id=state.document_id,
            related=[{"target_id": item["target_id"], "score": item["score"]} for item in related],
        )
        state.logs.append(f"linking: 已完成相似内容关联和双向链接建立，关联 {len(related)} 条。")
        return state
