from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import PipelineState
from app.services.vector_store import VectorStore


class LinkingAgent:
    def __init__(self, settings: Settings, repo: KnowledgeRepository, vector_store: VectorStore) -> None:
        self.settings = settings
        self.repo = repo
        self.vector_store = vector_store

    def run(self, state: PipelineState) -> PipelineState:
        self.vector_store.add_document(
            document_id=state.document_id,
            text=state.cleaned_text,
            metadata={"title": state.title, "category": state.category},
        )

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
