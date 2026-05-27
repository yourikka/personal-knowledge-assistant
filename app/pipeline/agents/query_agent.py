from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.services.openai_client import OpenAIService
from app.services.vector_store import VectorStore


class QueryAgent:
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

    def run(self, query: str, top_k: int, session_id: str | None = None) -> dict:
        logs = ["query: 已开始执行自然语言检索。"]
        ranked = self.vector_store.search(query=query, top_k=top_k)
        references = []
        for item in ranked:
            document = self.repo.get_document(item["id"])
            if not document:
                continue
            references.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "summary": document["summary"],
                    "category": document["category"],
                    "tags": document["tags"],
                    "source_uri": document["source_uri"],
                    "score": item["score"],
                }
            )

        history = self.repo.list_chat_turns(session_id) if session_id else []
        answer = self._compose_answer(query=query, references=references, history=history)
        logs.append("query: 已完成答案生成与引用拼装。")

        if session_id:
            self.repo.save_chat_turn(session_id, "user", query)
            self.repo.save_chat_turn(session_id, "assistant", answer)

        return {"answer": answer, "references": references, "logs": logs}

    def _compose_answer(self, query: str, references: list[dict], history: list[dict]) -> str:
        if not references:
            return f"当前知识库里没有找到和“{query}”足够相关的内容。建议先补充文档再检索。"

        if self.openai_service.enabled():
            refs_text = "\n".join(
                f"- 标题: {ref['title']}\n  摘要: {ref['summary']}\n  来源: {ref['source_uri']}"
                for ref in references
            )
            history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-4:])
            answer = self.openai_service.generate_text(
                system_prompt=(
                    "你是个人知识库问答助手。请只基于提供的引用内容回答，"
                    "输出简洁中文答案，并在答案中自然引用来源。"
                ),
                user_prompt=(
                    f"用户问题:\n{query}\n\n"
                    f"会话历史:\n{history_text or '无'}\n\n"
                    f"检索引用:\n{refs_text}"
                ),
            )
            if answer:
                return answer

        history_hint = ""
        if history:
            last_turn = history[-1]["content"][:60]
            history_hint = f"结合你上一轮上下文“{last_turn}”，"

        lead = f"{history_hint}我在知识库里找到 {len(references)} 条相关内容。"
        bullets = []
        for index, ref in enumerate(references, start=1):
            bullets.append(
                f"{index}. {ref['title']}：{ref['summary']} 来源：{ref['source_uri']}"
            )
        return lead + " " + " ".join(bullets)

