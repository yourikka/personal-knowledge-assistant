from __future__ import annotations

from app.models import PipelineState
from app.services.openai_client import OpenAIService
from app.services.self_check_service import SelfCheckService
from app.services.text_utils import summarize_text


class SummaryAgent:
    def __init__(self, openai_service: OpenAIService, self_check: SelfCheckService | None = None) -> None:
        self.openai_service = openai_service
        self.self_check = self_check

    def run(self, state: PipelineState) -> PipelineState:
        summary = summarize_text(state.cleaned_text, min_chars=100, max_chars=200)
        if self.openai_service.enabled():
            result = self.openai_service.generate_json(
                system_prompt=(
                    "你是个人知识库摘要助手，只能输出 JSON。"
                    "返回字段固定为 summary。"
                    "summary 必须是 100 到 200 字的中文摘要。"
                    "摘要要保留核心观点、关键结论和主要对象，不要寒暄，不要评价语，不要列标题，不要编造原文没有的信息。"
                    "如果原文信息不足，就基于已有内容尽量压缩，不要写套话。"
                ),
                user_prompt=(
                    f"标题: {state.title or '无'}\n"
                    f"分类: {state.category or '未分类'}\n"
                    f"标签: {', '.join(state.tags) if state.tags else '无'}\n"
                    f"正文:\n{state.cleaned_text[:5000]}"
                ),
            )
            if result and result.get("summary"):
                summary = str(result["summary"]).strip()
        if not summary:
            summary = state.cleaned_text[:180]
        if self.self_check:
            summary, check_logs = self.self_check.check_summary(summary, state.cleaned_text)
            state.logs.extend(check_logs)
        state.summary = summary
        state.logs.append(
            "summary: 已完成摘要提取。"
            + (" 使用 gpt-5.4。" if self.openai_service.enabled() else " 使用本地规则回退。")
        )
        return state
