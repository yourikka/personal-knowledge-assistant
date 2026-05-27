from __future__ import annotations

from app.services.openai_client import OpenAIService
from app.models import PipelineState
from app.services.text_utils import classify_text, extract_keywords


class ClassificationAgent:
    def __init__(self, openai_service: OpenAIService) -> None:
        self.openai_service = openai_service

    def run(self, state: PipelineState) -> PipelineState:
        category, confidence = classify_text(state.cleaned_text)
        tags = extract_keywords(state.cleaned_text, limit=5)
        if self.openai_service.enabled():
            result = self.openai_service.generate_json(
                system_prompt=(
                    "你是知识库分类与标签助手。请把内容分类到 技术/生活/学习 三类之一，"
                    "并返回 category、confidence、tags 字段。tags 为 3 到 5 个短标签。"
                ),
                user_prompt=state.cleaned_text[:4000],
            )
            if result:
                category = str(result.get("category") or category)
                confidence = float(result.get("confidence") or confidence)
                candidate_tags = result.get("tags") or tags
                if isinstance(candidate_tags, list):
                    tags = [str(item).strip() for item in candidate_tags if str(item).strip()][:5]
        state.category = category
        state.confidence = max(confidence, 0.7 if tags else confidence)
        state.tags = tags[:5]
        state.logs.append(
            "classification: 已完成主题分类与标签生成。"
            + (" 使用 gpt-5.4。" if self.openai_service.enabled() else " 使用本地规则回退。")
        )
        return state
