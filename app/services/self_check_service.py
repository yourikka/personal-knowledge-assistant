from __future__ import annotations

import re
from typing import Any

from app.config import Settings
from app.services.text_utils import extract_keywords, normalize_whitespace, summarize_text


class SelfCheckService:
    VALID_CATEGORIES = {"技术", "生活", "学习"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check_summary(self, summary: str, source_text: str) -> tuple[str, list[str]]:
        if not self.settings.self_check_enabled:
            return summary, []
        logs = []
        normalized = normalize_whitespace(summary)
        if len(normalized) > 200:
            normalized = normalized[:199].rstrip() + "…"
            logs.append("self_check.summary: 摘要超过 200 字，已截断。")
        if len(normalized) < 60 and len(source_text) >= 100:
            repaired = summarize_text(source_text, min_chars=100, max_chars=200)
            if len(repaired) > len(normalized):
                normalized = repaired
                logs.append("self_check.summary: 摘要过短，已用本地摘要规则重建。")
        return normalized, logs

    def check_classification(
        self,
        category: str,
        confidence: float,
        tags: list[str],
        source_text: str,
    ) -> tuple[str, float, list[str], list[str]]:
        if not self.settings.self_check_enabled:
            return category, confidence, tags, []
        logs = []
        if category not in self.VALID_CATEGORIES:
            category = "学习"
            confidence = min(confidence, 0.7)
            logs.append("self_check.classification: 分类不在允许集合，已回退为学习。")
        clean_tags = []
        seen = set()
        for tag in tags:
            normalized = str(tag).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                clean_tags.append(normalized)
        if len(clean_tags) < 3:
            for keyword in extract_keywords(source_text, limit=5):
                if keyword not in seen:
                    clean_tags.append(keyword)
                    seen.add(keyword)
                if len(clean_tags) >= 3:
                    break
            logs.append("self_check.classification: 标签少于 3 个，已补充关键词标签。")
        if clean_tags and confidence < 0.7:
            confidence = 0.7
            logs.append("self_check.classification: 有效标签存在但置信度过低，已提升到 0.7。")
        return category, min(1.0, max(0.0, confidence)), clean_tags[:5], logs

    def check_answer(self, answer: str, references: list[dict[str, Any]]) -> tuple[str, list[str]]:
        if not self.settings.self_check_enabled or not references:
            return answer, []
        logs = []
        valid_numbers = {str(index) for index in range(1, len(references) + 1)}
        cited_numbers = set(re.findall(r"\[(\d+)\]", answer))
        invalid = cited_numbers - valid_numbers
        repaired = answer
        for number in invalid:
            repaired = repaired.replace(f"[{number}]", "")
        if invalid:
            repaired = normalize_whitespace(repaired)
            logs.append("self_check.answer: 移除了不存在的引用编号。")
        if not (cited_numbers & valid_numbers):
            repaired = repaired.rstrip() + " [1]"
            logs.append("self_check.answer: 回答缺少文档引用，已补充首个引用。")
        return repaired, logs
