from __future__ import annotations

from app.models import PipelineState
from app.services.openai_client import OpenAIService
from app.services.self_check_service import SelfCheckService
from app.services.text_utils import STOPWORDS, classify_text, extract_keywords, normalize_whitespace, summarize_text


class MetadataAgent:
    def __init__(self, openai_service: OpenAIService, self_check: SelfCheckService | None = None) -> None:
        self.openai_service = openai_service
        self.self_check = self_check

    def run(self, state: PipelineState, use_model: bool = True) -> PipelineState:
        category, confidence = classify_text(state.cleaned_text)
        tags = self._local_tags(state)
        summary = self._local_summary(state.cleaned_text)
        source = "本地规则回退"

        if use_model and self.openai_service.enabled():
            try:
                result = self.openai_service.generate_json(
                    system_prompt=(
                        "你是个人知识库元数据助手，只能输出 JSON。"
                        "返回字段固定为 category、confidence、tags、summary。"
                        "category 必须是 技术、生活、学习 三类之一。"
                        "confidence 是 0 到 1 的数字。"
                        "tags 必须是 3 到 5 个具体、可检索、非重复短标签。"
                        "summary 必须是 100 到 200 字中文摘要，保留核心观点、关键结论和主要对象。"
                        "不要寒暄，不要评价语，不要列标题，不要编造原文没有的信息。"
                    ),
                    user_prompt=(
                        f"标题: {state.title or '无'}\n"
                        f"来源类型: {state.request.source_type}\n"
                        f"正文:\n{state.cleaned_text[:5000]}"
                    ),
                )
            except Exception as error:
                result = None
                state.logs.append(f"metadata: 模型生成失败，已回退本地规则：{error}")
            if result:
                category = str(result.get("category") or category)
                confidence = self._float_or_default(result.get("confidence"), confidence)
                candidate_tags = result.get("tags") or tags
                if isinstance(candidate_tags, list):
                    tags = [str(item).strip() for item in candidate_tags if str(item).strip()][:5]
                if result.get("summary"):
                    summary = str(result["summary"]).strip()
                source = "gpt-5.4"
            else:
                state.logs.append("metadata: 模型未返回可用 JSON，已使用本地规则。")

        if not summary:
            summary = normalize_whitespace(state.cleaned_text)[:180]
        confidence = max(confidence, 0.7 if tags else confidence)
        if self.self_check:
            category, confidence, tags, class_logs = self.self_check.check_classification(
                category=category,
                confidence=confidence,
                tags=tags,
                source_text=state.cleaned_text,
            )
            summary, summary_logs = self.self_check.check_summary(summary, state.cleaned_text)
            state.logs.extend([*class_logs, *summary_logs])

        state.category = category
        state.confidence = confidence
        state.tags = tags[:5]
        state.summary = summary
        state.logs.append(f"classification: 已完成主题分类与标签生成。 使用{source}。")
        state.logs.append(f"summary: 已完成摘要提取。 使用{source}。")
        state.logs.append(f"metadata: 已完成分类、标签和摘要生成。 使用{source}。")
        return state

    def run_local(self, state: PipelineState) -> PipelineState:
        return self.run(state, use_model=False)

    def _float_or_default(self, value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _local_tags(self, state: PipelineState) -> list[str]:
        candidates = [
            *extract_keywords(f"{state.title}\n{state.cleaned_text}", limit=12),
            *extract_keywords(state.cleaned_text, limit=12),
        ]
        tags: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            tag = self._normalize_tag(item)
            key = tag.lower()
            if not tag or key in seen:
                continue
            seen.add(key)
            tags.append(tag)
            if len(tags) >= 5:
                break
        return tags

    def _normalize_tag(self, value: str) -> str:
        tag = normalize_whitespace(value).strip(" ，。；;：:（）()[]【】#")
        if not tag:
            return ""
        lowered = tag.lower()
        if lowered in STOPWORDS:
            return ""
        if len(tag) < 2 or len(tag) > 18:
            return ""
        if tag.isascii() and len(tag) < 4:
            return ""
        if any(fragment in tag for fragment in ("可以", "如果", "这个", "那个")):
            return ""
        return tag

    def _local_summary(self, text: str) -> str:
        summary = summarize_text(text, min_chars=100, max_chars=200)
        if summary:
            return summary
        return normalize_whitespace(text)[:180]
