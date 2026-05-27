from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.services.openai_client import OpenAIService
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore


class QueryAgent:
    def __init__(
        self,
        settings: Settings,
        repo: KnowledgeRepository,
        vector_store: VectorStore,
        openai_service: OpenAIService,
        rag_service: RAGService | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.vector_store = vector_store
        self.openai_service = openai_service
        self.rag_service = rag_service or RAGService(settings, repo, vector_store, openai_service)

    def run(self, query: str, top_k: int, session_id: str | None = None) -> dict:
        logs = ["query: 已开始执行 RAG 检索。"]
        history = self.repo.list_chat_turns(session_id) if session_id else []
        retrieval = self.rag_service.retrieve(query=query, top_k=top_k, session_id=session_id)
        references = retrieval["references"]
        answer = self._compose_answer(
            query=query,
            references=references,
            context=retrieval["context"],
            expanded_queries=retrieval["expanded_queries"],
            history=history,
        )
        logs.extend(retrieval["logs"])
        logs.append("query: 已完成 RAG 答案生成与引用拼装。")

        if session_id:
            self.repo.save_chat_turn(session_id, "user", query)
            self.repo.save_chat_turn(session_id, "assistant", answer)

        return {"answer": answer, "references": references, "logs": logs}

    def _compose_answer(
        self,
        query: str,
        references: list[dict],
        context: str,
        expanded_queries: list[str],
        history: list[dict],
    ) -> str:
        if not references:
            return f"当前知识库里没有找到和“{query}”足够相关的内容。建议先补充文档再检索。"

        if self.openai_service.enabled():
            history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-4:])
            answer = self.openai_service.generate_text(
                system_prompt=(
                    "你是个人知识库 RAG 问答助手。必须只基于给定上下文回答。"
                    "如果上下文不足，请明确说不足。答案要简洁，并用 [1]、[2] 这样的编号引用来源。"
                ),
                user_prompt=(
                    f"用户问题:\n{query}\n\n"
                    f"查询改写:\n{', '.join(expanded_queries)}\n\n"
                    f"会话历史:\n{history_text or '无'}\n\n"
                    f"RAG 上下文:\n{context}"
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
                f"[{index}] {ref['title']}：{ref['summary']} 来源：{ref['source_uri']}"
            )
        return lead + " " + " ".join(bullets)
