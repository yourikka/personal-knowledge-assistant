from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.db import KnowledgeRepository
from app.services.openai_client import OpenAIService
from app.services.text_utils import extract_keywords, overlap_score, tokenize
from app.services.vector_store import VectorStore


class RAGService:
    def __init__(
        self,
        settings: Settings,
        repo: KnowledgeRepository,
        vector_store: VectorStore,
        openai_service: OpenAIService,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.vector_store = vector_store
        self.openai_service = openai_service

    def retrieve(
        self,
        query: str,
        top_k: int,
        session_id: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        logs: list[str] = []
        expanded_queries = self.expand_query(query=query, session_id=session_id)
        logs.append(f"rag: 查询扩展 {len(expanded_queries)} 条。")

        candidate_limit = max(top_k * self.settings.rag_candidate_multiplier, top_k)
        candidates = self._collect_candidates(expanded_queries, candidate_limit, exclude_ids or set())
        logs.append(f"rag: 多路召回候选 {len(candidates)} 条。")

        ranked = self._rerank(query=query, candidates=candidates)
        filtered = [item for item in ranked if item["score"] >= self.settings.rag_min_score]
        selected = self._mmr_select(query=query, candidates=filtered, top_k=top_k)
        references, context = self._build_context(selected)
        logs.append(f"rag: 重排后选中 {len(references)} 条，上下文 {len(context)} 字符。")

        return {
            "query": query,
            "expanded_queries": expanded_queries,
            "references": references,
            "context": context,
            "logs": logs,
        }

    def expand_query(self, query: str, session_id: str | None = None) -> list[str]:
        queries = [query]
        history = self.repo.list_chat_turns(session_id, limit=4) if session_id else []
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-4:])

        if self.settings.rag_rewrite_enabled and self.openai_service.enabled():
            result = self.openai_service.generate_json(
                system_prompt=(
                    "你是 RAG 查询改写器。请返回 JSON，字段 queries 是 2 到 4 个中文查询，"
                    "覆盖用户原意、关键词表达和更具体的检索表达。不要添加用户没有问的主题。"
                ),
                user_prompt=f"用户问题:\n{query}\n\n会话历史:\n{history_text or '无'}",
            )
            if result and isinstance(result.get("queries"), list):
                for item in result["queries"]:
                    text = str(item).strip()
                    if text and text not in queries:
                        queries.append(text)

        for keyword in extract_keywords(query, limit=4):
            if keyword not in queries:
                queries.append(keyword)

        return queries[: max(1, self.settings.rag_multi_query_limit)]

    def _collect_candidates(
        self,
        queries: list[str],
        candidate_limit: int,
        exclude_ids: set[str],
    ) -> dict[str, dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        for query_index, query in enumerate(queries):
            vector_hits = self.vector_store.search(query=query, top_k=candidate_limit * 4, kind="chunk")
            for rank, hit in enumerate(vector_hits):
                chunk = self.repo.get_chunk(hit["id"])
                if not chunk or chunk["document_id"] in exclude_ids:
                    continue
                document = self.repo.get_document(chunk["document_id"])
                if not document:
                    continue
                self._add_candidate(
                    candidates=candidates,
                    chunk=chunk,
                    document=document,
                    signal={"source": "chunk_vector", "query": query, "score": hit["score"], "rank": rank + 1},
                )

            keyword_hits = self.repo.search_chunks_keyword(query, limit=candidate_limit, exclude_document_ids=exclude_ids)
            for rank, chunk in enumerate(keyword_hits):
                document = self.repo.get_document(chunk["document_id"])
                if not document:
                    continue
                keyword_score = 1.0 / (rank + 1 + query_index)
                self._add_candidate(
                    candidates=candidates,
                    chunk=chunk,
                    document=document,
                    signal={"source": "chunk_keyword", "query": query, "score": keyword_score, "rank": rank + 1},
                )
            if self.settings.rag_hierarchical_enabled:
                self._collect_hierarchical_candidates(
                    query=query,
                    candidate_limit=candidate_limit,
                    exclude_ids=exclude_ids,
                    candidates=candidates,
                )
        return candidates

    def _collect_hierarchical_candidates(
        self,
        query: str,
        candidate_limit: int,
        exclude_ids: set[str],
        candidates: dict[str, dict[str, Any]],
    ) -> None:
        document_hits = self.vector_store.search(
            query=query,
            top_k=min(self.settings.rag_document_top_k, candidate_limit),
            kind="document",
            exclude_ids=exclude_ids,
        )
        for rank, hit in enumerate(document_hits):
            document = self.repo.get_document(hit["id"])
            if not document:
                continue
            for chunk in self._best_chunks_for_document(query, document["id"], self.settings.rag_section_chunk_limit):
                self._add_candidate(
                    candidates=candidates,
                    chunk=chunk,
                    document=document,
                    signal={"source": "document_vector", "query": query, "score": hit["score"], "rank": rank + 1},
                )

        section_hits = self.vector_store.search(
            query=query,
            top_k=min(self.settings.rag_section_top_k, candidate_limit * 2),
            kind="section",
        )
        section_hits.extend(
            {
                "id": section["id"],
                "score": 0.45,
                "source": "section_keyword",
            }
            for section in self.repo.search_sections_keyword(
                query=query,
                limit=min(self.settings.rag_section_top_k, candidate_limit * 2),
                exclude_document_ids=exclude_ids,
            )
        )
        seen_sections: set[str] = set()
        for rank, hit in enumerate(section_hits):
            section_id = hit["id"]
            if section_id in seen_sections:
                continue
            seen_sections.add(section_id)
            section = self.repo.get_section(section_id)
            if not section or section["document_id"] in exclude_ids:
                continue
            document = self.repo.get_document(section["document_id"])
            if not document:
                continue
            source = hit.get("source", "section_vector")
            for chunk in self._chunks_for_section(section)[: self.settings.rag_section_chunk_limit]:
                self._add_candidate(
                    candidates=candidates,
                    chunk=chunk,
                    document=document,
                    signal={"source": source, "query": query, "score": hit["score"], "rank": rank + 1},
                )

    def _add_candidate(
        self,
        candidates: dict[str, dict[str, Any]],
        chunk: dict[str, Any],
        document: dict[str, Any],
        signal: dict[str, Any],
    ) -> None:
        entry = candidates.setdefault(
            chunk["id"],
            {
                "chunk": chunk,
                "document": document,
                "signals": [],
                "best_rank": signal["rank"],
                "raw_score": 0.0,
            },
        )
        entry["signals"].append(signal)
        entry["best_rank"] = min(entry["best_rank"], signal["rank"])
        entry["raw_score"] = max(entry["raw_score"], float(signal["score"]))

    def _best_chunks_for_document(self, query: str, document_id: str, limit: int) -> list[dict[str, Any]]:
        chunks = self.repo.list_document_chunks(document_id)
        ranked = sorted(
            chunks,
            key=lambda chunk: self.vector_store.similarity(query, chunk["text"]),
            reverse=True,
        )
        return ranked[:limit]

    def _chunks_for_section(self, section: dict[str, Any]) -> list[dict[str, Any]]:
        chunk_ids = section.get("metadata", {}).get("chunk_ids") or []
        chunks = [chunk for chunk_id in chunk_ids if (chunk := self.repo.get_chunk(chunk_id))]
        if chunks:
            return chunks

        return [
            chunk
            for chunk in self.repo.list_document_chunks(section["document_id"])
            if chunk["char_start"] < section["char_end"] and chunk["char_end"] > section["char_start"]
        ]

    def _rerank(self, query: str, candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        query_tags = set(tokenize(query))
        ranked = []
        for chunk_id, entry in candidates.items():
            chunk = entry["chunk"]
            document = entry["document"]
            text = self._chunk_text(document, chunk)
            semantic_score = self.vector_store.similarity(query, text)
            lexical_score = overlap_score(query, text)
            tag_score = self._tag_score(query_tags, document.get("tags", []))
            recent_score = self._recent_score(document.get("created_at", ""))
            signal_score = self._signal_score(entry["signals"])

            score = (
                semantic_score * 0.46
                + lexical_score * 0.22
                + signal_score * 0.18
                + tag_score * self.settings.rag_tag_boost
                + recent_score * self.settings.rag_recent_boost
            )
            ranked.append(
                {
                    "id": chunk_id,
                    "chunk": chunk,
                    "document": document,
                    "score": round(score, 4),
                    "semantic_score": semantic_score,
                    "lexical_score": round(lexical_score, 4),
                    "tag_score": round(tag_score, 4),
                    "recent_score": round(recent_score, 4),
                    "signals": entry["signals"],
                    "text": text,
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _mmr_select(self, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        remaining = candidates[:]
        while remaining and len(selected) < top_k:
            best_item = None
            best_score = -math.inf
            for item in remaining:
                redundancy = 0.0
                if selected:
                    redundancy = max(self.vector_store.similarity(item["text"], chosen["text"]) for chosen in selected)
                mmr_score = self.settings.rag_mmr_lambda * item["score"] - (1 - self.settings.rag_mmr_lambda) * redundancy
                if mmr_score > best_score:
                    best_item = item
                    best_score = mmr_score
            if best_item is None:
                break
            best_item["mmr_score"] = round(best_score, 4)
            selected.append(best_item)
            remaining = [item for item in remaining if item["id"] != best_item["id"]]
        return selected

    def _build_context(self, selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        references = []
        context_blocks = []
        budget = self.settings.rag_context_char_budget
        used = 0
        for index, item in enumerate(selected, start=1):
            chunk = item["chunk"]
            document = item["document"]
            heading_path = chunk.get("metadata", {}).get("heading_path") or []
            heading = " > ".join(heading_path) if heading_path else "无"
            block = (
                f"[{index}] {document['title']} (chunk #{chunk['chunk_index']})\n"
                f"来源: {document['source_uri']}\n"
                f"分类: {document['category']}\n"
                f"标签: {', '.join(document['tags'])}\n"
                f"标题路径: {heading}\n"
                f"摘要: {document['summary']}\n"
                f"原文片段: {chunk['text']}"
            )
            if used + len(block) > budget:
                block = block[: max(0, budget - used)]
            if not block:
                break
            context_blocks.append(block)
            used += len(block)
            references.append(
                {
                    "id": document["id"],
                    "chunk_id": chunk["id"],
                    "chunk_index": chunk["chunk_index"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                    "heading_path": heading_path,
                    "title": document["title"],
                    "summary": document["summary"],
                    "category": document["category"],
                    "tags": document["tags"],
                    "source_uri": document["source_uri"],
                    "score": item["score"],
                    "mmr_score": item.get("mmr_score", item["score"]),
                    "signals": item["signals"][:5],
                }
            )
            if used >= budget:
                break
        return references, "\n\n".join(context_blocks)

    def _chunk_text(self, document: dict[str, Any], chunk: dict[str, Any]) -> str:
        heading_path = chunk.get("metadata", {}).get("heading_path") or []
        return " ".join(
            [
                document.get("title", ""),
                document.get("summary", ""),
                " ".join(document.get("tags", [])),
                " ".join(str(item) for item in heading_path),
                chunk.get("text", ""),
            ]
        )

    def _tag_score(self, query_tokens: set[str], tags: list[str]) -> float:
        tag_tokens = set(token for tag in tags for token in tokenize(tag))
        if not query_tokens or not tag_tokens:
            return 0.0
        return len(query_tokens & tag_tokens) / len(query_tokens | tag_tokens)

    def _recent_score(self, created_at: str) -> float:
        try:
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except ValueError:
            return 0.0
        age_days = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86400)
        return 1.0 / (1.0 + age_days / 30)

    def _signal_score(self, signals: list[dict[str, Any]]) -> float:
        if not signals:
            return 0.0
        weighted = 0.0
        total = 0.0
        for signal in signals:
            weight = 1.0 if "vector" in signal["source"] else 0.85
            weighted += float(signal["score"]) * weight
            total += weight
        return min(1.0, weighted / max(total, 1e-9))
