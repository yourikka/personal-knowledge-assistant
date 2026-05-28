from __future__ import annotations

from app.services.openai_client import OpenAIService
from app.models import PipelineState
from app.services.self_check_service import SelfCheckService
from app.services.text_utils import classify_text, extract_keywords


class ClassificationAgent:
    def __init__(self, openai_service: OpenAIService, self_check: SelfCheckService | None = None) -> None:
        self.openai_service = openai_service
        self.self_check = self_check

    def run(self, state: PipelineState) -> PipelineState:
        category, confidence = classify_text(state.cleaned_text)
        tags = extract_keywords(state.cleaned_text, limit=5)
        if self.openai_service.enabled():
            result = self.openai_service.generate_json(
                system_prompt=(
                    "你是个人知识库的分类与标签助手，只能输出 JSON。"
                    "你必须把内容分类到 技术、生活、学习 三类之一。"
                    "返回字段固定为 category、confidence、tags。"
                    "category 必须是这三类之一；confidence 是 0 到 1 的数字；"
                    "tags 必须是 3 到 5 个短标签。"
                    "标签要具体、可检索、避免空泛词，不要重复，不要输出句子。"
                    "如果内容跨多个主题，选择主主题而不是多选。"
                ),
                user_prompt=(
                    f"标题: {state.title or '无'}\n"
                    f"来源类型: {state.request.source_type}\n"
                    f"正文:\n{state.cleaned_text[:4000]}"
                ),
            )
            if result:
                category = str(result.get("category") or category)
                confidence = float(result.get("confidence") or confidence)
                candidate_tags = result.get("tags") or tags
                if isinstance(candidate_tags, list):
                    tags = [str(item).strip() for item in candidate_tags if str(item).strip()][:5]
        confidence = max(confidence, 0.7 if tags else confidence)
        if self.self_check:
            category, confidence, tags, check_logs = self.self_check.check_classification(
                category=category,
                confidence=confidence,
                tags=tags,
                source_text=state.cleaned_text,
            )
            state.logs.extend(check_logs)
        state.category = category
        state.confidence = confidence
        state.tags = tags[:5]
        state.logs.append(
            "classification: 已完成主题分类与标签生成。"
            + (" 使用 gpt-5.4。" if self.openai_service.enabled() else " 使用本地规则回退。")
        )
        return state
