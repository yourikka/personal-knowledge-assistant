from __future__ import annotations

from typing import Any

from app.config import Settings
from app.db import KnowledgeRepository
from app.services.text_utils import extract_keywords, tokenize


class PersonalizationService:
    def __init__(self, settings: Settings, repo: KnowledgeRepository) -> None:
        self.settings = settings
        self.repo = repo

    def learn_query(self, session_id: str | None, query: str) -> None:
        if not session_id:
            return
        self.repo.record_query_profile(session_id, extract_keywords(query, limit=6), weight=1.0)

    def record_click(self, session_id: str, document_id: str, query: str) -> None:
        self.repo.record_document_click(session_id=session_id, document_id=document_id, query=query)

    def record_feedback(
        self,
        session_id: str,
        query: str,
        rating: int,
        document_id: str | None = None,
        comment: str | None = None,
    ) -> None:
        self.repo.record_query_feedback(
            session_id=session_id,
            query=query,
            rating=rating,
            document_id=document_id,
            comment=comment,
        )

    def score_document(self, session_id: str | None, document: dict[str, Any]) -> float:
        if not session_id:
            return 0.0
        profile = self.repo.list_query_profile(session_id, limit=20)
        clicks = self.repo.document_click_counts(session_id)
        if not profile and not clicks:
            return 0.0

        doc_tokens = set(tokenize(" ".join([document.get("title", ""), document.get("summary", ""), *document.get("tags", [])])))
        total_weight = sum(float(item["weight"]) for item in profile) or 1.0
        profile_score = 0.0
        for item in profile:
            tag = item["tag"]
            if tag in doc_tokens or tag in document.get("title", "").lower() or tag in document.get("summary", "").lower():
                profile_score += float(item["weight"]) / total_weight

        click_score = min(1.0, clicks.get(document["id"], 0) / 5)
        return min(1.0, profile_score * 0.65 + click_score * 0.35)
