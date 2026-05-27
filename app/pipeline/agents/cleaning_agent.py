from __future__ import annotations

from app.models import PipelineState
from app.services.text_utils import fix_mojibake, normalize_document_text, remove_noise, text_stats


class CleaningAgent:
    def run(self, state: PipelineState) -> PipelineState:
        before_stats = text_stats(state.parsed_text)
        cleaned = fix_mojibake(state.parsed_text)
        cleaned = remove_noise(cleaned)
        cleaned = normalize_document_text(cleaned)
        after_stats = text_stats(cleaned)
        state.cleaned_text = cleaned
        state.metadata["cleaning"] = {
            "before": before_stats,
            "after": after_stats,
            "removed_chars": max(0, before_stats["chars"] - after_stats["chars"]),
        }
        state.logs.append(
            f"cleaning: 已完成文本清洗、去噪和统一格式，字符数 {before_stats['chars']} -> {after_stats['chars']}。"
        )
        return state
