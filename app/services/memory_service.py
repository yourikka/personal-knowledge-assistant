from __future__ import annotations

import hashlib
from typing import Any

from app.config import Settings
from app.db import KnowledgeRepository
from app.services.openai_client import OpenAIService
from app.services.text_utils import extract_keywords, overlap_score
from app.services.vector_store import VectorStore


class MemoryService:
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

    def bootstrap(self) -> None:
        if not self.settings.memory_enabled:
            return
        for memory in self.repo.list_memories(limit=self.settings.memory_bootstrap_limit):
            self._index_memory(memory)

    def retrieve(self, query: str, session_id: str | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
        if not self.settings.memory_enabled:
            return []

        limit = top_k or self.settings.memory_top_k
        candidates: dict[str, dict[str, Any]] = {}

        for hit in self.vector_store.search(query=query, top_k=limit * 4, kind="memory"):
            memory = self.repo.get_memory(hit["id"])
            if not memory or not self._visible_in_session(memory, session_id):
                continue
            score = self._score_memory(query=query, memory=memory, retrieval_score=float(hit["score"]))
            if score >= self.settings.memory_min_score:
                candidates[memory["id"]] = {**memory, "score": round(score, 4), "signal": "vector"}

        for memory in self.repo.search_memories_keyword(query=query, session_id=session_id, limit=limit * 2):
            score = self._score_memory(query=query, memory=memory, retrieval_score=0.35)
            current = candidates.get(memory["id"])
            if score >= self.settings.memory_min_score and (current is None or score > current["score"]):
                candidates[memory["id"]] = {**memory, "score": round(score, 4), "signal": "keyword"}

        ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    def format_context(self, memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "无"
        blocks = []
        for index, memory in enumerate(memories, start=1):
            tags = ", ".join(memory.get("tags", [])) or "无"
            scope = "全局" if memory.get("session_id") is None else f"会话 {memory['session_id']}"
            blocks.append(
                f"[M{index}] 类型: {memory['kind']}\n"
                f"范围: {scope}\n"
                f"重要性: {memory['importance']:.2f}\n"
                f"标签: {tags}\n"
                f"内容: {memory['content']}"
            )
        return "\n\n".join(blocks)

    def learn_from_turn(
        self,
        query: str,
        answer: str,
        session_id: str | None,
        references: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.settings.memory_enabled or not session_id:
            return []

        if self.openai_service.enabled():
            learned = self._extract_with_model(query=query, answer=answer, session_id=session_id, references=references)
        else:
            learned = self._extract_with_rules(query=query, session_id=session_id)

        stored = []
        for memory in learned[: self.settings.memory_write_limit]:
            content = str(memory.get("content") or "").strip()
            if not content:
                continue
            kind = self._normalize_kind(str(memory.get("kind") or "fact"))
            importance = self._normalize_importance(memory.get("importance", 0.5))
            tags = memory.get("tags") or extract_keywords(content, limit=5)
            if not isinstance(tags, list):
                tags = extract_keywords(content, limit=5)
            record = {
                "id": self._memory_id(session_id=session_id, kind=kind, content=content),
                "session_id": session_id,
                "kind": kind,
                "content": content[: self.settings.memory_max_content_chars],
                "importance": importance,
                "tags": [str(item).strip() for item in tags if str(item).strip()][:5],
                "metadata": {
                    "source": "chat_turn",
                    "reference_ids": [item.get("id") for item in references[:5] if item.get("id")],
                },
            }
            self.repo.upsert_memory(record)
            self._index_memory(record)
            stored.append(record)
        return stored

    def _extract_with_model(
        self,
        query: str,
        answer: str,
        session_id: str,
        references: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        reference_text = "\n".join(
            f"- {item.get('title', '')}: {item.get('summary', '')}" for item in references[:5]
        )
        try:
            result = self.openai_service.generate_json(
                system_prompt=(
                    "你是个人知识库的记忆提取器，只能输出 JSON。"
                    "返回字段固定为 memories，值是数组。"
                    "只提取未来问答中值得复用的稳定信息，包括用户偏好、长期目标、项目决策、已确认事实。"
                    "不要保存临时问题、寒暄、一次性命令、模型回答格式要求、引用编号或无法确认的猜测。"
                    "每条记忆包含 content、kind、importance、tags。"
                    "kind 只能是 preference、goal、decision、fact、project。"
                    "importance 是 0 到 1 的数字。tags 是 1 到 5 个短标签。"
                    "最多返回 3 条；没有值得保存的信息时返回空数组。"
                ),
                user_prompt=(
                    f"会话 ID: {session_id}\n\n"
                    f"用户问题:\n{query}\n\n"
                    f"助手回答:\n{answer[:2500]}\n\n"
                    f"引用资料:\n{reference_text or '无'}"
                ),
            )
        except Exception:
            return []
        memories = result.get("memories") if result else []
        return memories if isinstance(memories, list) else []

    def _extract_with_rules(self, query: str, session_id: str) -> list[dict[str, Any]]:
        triggers = ("记住", "以后", "我的偏好", "我希望", "我喜欢", "我不喜欢", "默认")
        if not any(trigger in query for trigger in triggers):
            return []
        return [
            {
                "content": query[: self.settings.memory_max_content_chars],
                "kind": "preference",
                "importance": 0.72,
                "tags": extract_keywords(query, limit=5),
            }
        ]

    def _index_memory(self, memory: dict[str, Any]) -> None:
        self.vector_store.add_memory(
            memory_id=memory["id"],
            text=memory["content"],
            metadata={
                "session_id": memory.get("session_id") or "",
                "kind": "memory",
                "memory_kind": memory["kind"],
                "importance": float(memory.get("importance", 0.5)),
            },
        )

    def _score_memory(self, query: str, memory: dict[str, Any], retrieval_score: float) -> float:
        lexical = overlap_score(query, memory["content"])
        importance = float(memory.get("importance", 0.5))
        return retrieval_score * 0.62 + lexical * 0.18 + importance * 0.20

    def _visible_in_session(self, memory: dict[str, Any], session_id: str | None) -> bool:
        memory_session = memory.get("session_id")
        return memory_session is None or bool(session_id and memory_session == session_id)

    def _memory_id(self, session_id: str, kind: str, content: str) -> str:
        digest = hashlib.sha256(f"{session_id}:{kind}:{content.strip()}".encode("utf-8")).hexdigest()[:24]
        return f"mem-{digest}"

    def _normalize_kind(self, value: str) -> str:
        allowed = {"preference", "goal", "decision", "fact", "project"}
        return value if value in allowed else "fact"

    def _normalize_importance(self, value: Any) -> float:
        try:
            importance = float(value)
        except (TypeError, ValueError):
            importance = 0.5
        return min(1.0, max(0.0, importance))
