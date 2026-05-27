from __future__ import annotations

from app.models import PipelineState
from app.services.openai_client import OpenAIService
from app.services.text_utils import summarize_text


class SummaryAgent:
    def __init__(self, openai_service: OpenAIService) -> None:
        self.openai_service = openai_service

    def run(self, state: PipelineState) -> PipelineState:
        summary = summarize_text(state.cleaned_text, min_chars=100, max_chars=200)
        if self.openai_service.enabled():
            result = self.openai_service.generate_json(
                system_prompt=(
                    "你是知识库摘要助手。请输出 JSON，字段为 summary。"
                    "要求 100 到 200 字，保留核心观点，不要废话。"
                ),
                user_prompt=state.cleaned_text[:5000],
            )
            if result and result.get("summary"):
                summary = str(result["summary"]).strip()
        if not summary:
            summary = state.cleaned_text[:180]
        state.summary = summary
        state.logs.append(
            "summary: 已完成摘要提取。"
            + (" 使用 gpt-5.4。" if self.openai_service.enabled() else " 使用本地规则回退。")
        )
        return state
