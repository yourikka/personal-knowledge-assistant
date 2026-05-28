from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.services.memory_service import MemoryService
from app.services.openai_client import OpenAIService
from app.services.rag_service import RAGService
from app.services.self_check_service import SelfCheckService
from app.services.vector_store import VectorStore


class QueryAgent:
    def __init__(
        self,
        settings: Settings,
        repo: KnowledgeRepository,
        vector_store: VectorStore,
        openai_service: OpenAIService,
        rag_service: RAGService | None = None,
        memory_service: MemoryService | None = None,
        self_check: SelfCheckService | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.vector_store = vector_store
        self.openai_service = openai_service
        self.rag_service = rag_service or RAGService(settings, repo, vector_store, openai_service)
        self.memory_service = memory_service
        self.self_check = self_check

    def run(self, query: str, top_k: int, session_id: str | None = None) -> dict:
        logs = ["query: 已开始执行 RAG 检索。"]
        history = self.repo.list_chat_turns(session_id) if session_id else []
        memories = self.memory_service.retrieve(query=query, session_id=session_id) if self.memory_service else []
        if self.memory_service:
            logs.append(f"memory: 已召回 {len(memories)} 条相关记忆。")
        retrieval = self.rag_service.retrieve(query=query, top_k=top_k, session_id=session_id)
        references = retrieval["references"]
        answer = self._compose_answer(
            query=query,
            references=references,
            context=retrieval["context"],
            expanded_queries=retrieval["expanded_queries"],
            history=history,
            memories=memories,
        )
        if self.self_check:
            answer, check_logs = self.self_check.check_answer(answer, references)
            logs.extend(check_logs)
        logs.extend(retrieval["logs"])
        logs.append("query: 已完成 RAG 答案生成与引用拼装。")

        if session_id:
            self.repo.save_chat_turn(session_id, "user", query)
            self.repo.save_chat_turn(session_id, "assistant", answer)
            if self.memory_service:
                learned = self.memory_service.learn_from_turn(
                    query=query,
                    answer=answer,
                    session_id=session_id,
                    references=references,
                )
                logs.append(f"memory: 已写入 {len(learned)} 条新记忆。")

        return {"answer": answer, "references": references, "memories": memories, "logs": logs}

    def _compose_answer(
        self,
        query: str,
        references: list[dict],
        context: str,
        expanded_queries: list[str],
        history: list[dict],
        memories: list[dict],
    ) -> str:
        if not references:
            if memories:
                return self._compose_memory_only_answer(query=query, memories=memories, history=history)
            return f"当前知识库里没有找到和“{query}”足够相关的内容。建议先补充文档再检索。"

        if self.openai_service.enabled():
            history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-4:])
            memory_context = self.memory_service.format_context(memories) if self.memory_service else "无"
            answer = self.openai_service.generate_text(
                system_prompt=(
                    "你是个人知识库 RAG 问答助手。"
                    "你只能依据提供的 RAG 上下文、相关记忆和会话历史回答，不能补充这些材料里没有的事实。"
                    "如果证据不足、上下文没有答案、或结论存在歧义，必须明确说明“当前资料不足以确认”。"
                    "相关记忆只能用于理解用户偏好、长期目标和已确认背景，不能替代文档证据。"
                    "回答时优先直接给结论，再补充 2 到 4 条关键依据。"
                    "凡是使用了上下文中的事实、数字、判断、时间、来源，都必须在对应句子末尾标注引用编号，如 [1] 或 [1][2]。"
                    "引用编号必须严格对应给定上下文里的编号，不能编造不存在的编号，不能遗漏关键结论的引用。"
                    "记忆编号如 [M1] 只能在说明用户偏好或项目背景时使用，不能作为文档事实引用。"
                    "如果多个来源支持同一结论，可以并列引用。"
                    "不要输出与问题无关的铺垫，不要使用“根据常识”“一般来说”这类脱离上下文的表述。"
                ),
                user_prompt=(
                    "请基于以下材料作答。\n\n"
                    f"用户问题:\n{query}\n\n"
                    f"查询改写:\n{', '.join(expanded_queries) or query}\n\n"
                    f"会话历史:\n{history_text or '无'}\n\n"
                    f"相关记忆:\n{memory_context}\n\n"
                    "回答要求:\n"
                    "1. 只使用下方 RAG 上下文。\n"
                    "2. 如果无法从上下文得出答案，直接说明当前资料不足以确认。\n"
                    "3. 引用格式只能是 [1]、[2] 这种编号，且必须和上下文编号一致。\n"
                    "4. 相关记忆只能辅助理解用户意图；涉及资料结论时仍必须引用 RAG 上下文。\n"
                    "5. 优先简洁中文回答；必要时用短小分点，不要长篇复述原文。\n\n"
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

    def _compose_memory_only_answer(self, query: str, memories: list[dict], history: list[dict]) -> str:
        if self.openai_service.enabled() and self.memory_service:
            history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-4:])
            answer = self.openai_service.generate_text(
                system_prompt=(
                    "你是个人知识库记忆问答助手。"
                    "当前没有可用文档引用，只能基于相关记忆和会话历史回答。"
                    "如果相关记忆不足以回答，必须明确说明“当前记忆不足以确认”。"
                    "使用记忆内容时，在对应句子末尾标注记忆编号，如 [M1]。"
                    "不要把记忆说成文档证据，不要编造没有出现过的信息。"
                ),
                user_prompt=(
                    f"用户问题:\n{query}\n\n"
                    f"会话历史:\n{history_text or '无'}\n\n"
                    f"相关记忆:\n{self.memory_service.format_context(memories)}"
                ),
            )
            if answer:
                return answer

        bullets = []
        for index, memory in enumerate(memories, start=1):
            bullets.append(f"[M{index}] {memory['content']}")
        return "当前没有找到相关文档，但命中了以下记忆：" + " ".join(bullets)
