from __future__ import annotations

import os
import re

from app.models import PipelineState
from app.services.parser_utils import parse_content
from app.services.text_utils import normalize_whitespace


class ParserAgent:
    def run(self, state: PipelineState) -> PipelineState:
        parsed_text, metadata = parse_content(
            source_type=state.request.source_type,
            raw_bytes=state.raw_bytes,
            metadata=state.metadata,
        )
        if not parsed_text.strip():
            raise ValueError("解析结果为空，无法入库。")
        state.parsed_text = parsed_text
        state.metadata = metadata
        state.title = self._select_title(state, metadata)
        state.logs.append(f"parser: 已完成格式解析和元数据提取，提取 {len(parsed_text)} 字符。")
        return state

    def _select_title(self, state: PipelineState, metadata: dict) -> str:
        candidates = [
            state.request.title,
            metadata.get("detected_title"),
            metadata.get("filename"),
            state.source_uri,
        ]
        for candidate in candidates:
            title = self._clean_title(str(candidate or ""))
            if title:
                return title[:120]
        return "未命名文档"

    def _clean_title(self, value: str) -> str:
        title = os.path.basename(value.strip()) if value.startswith(("/", ".")) else value.strip()
        title = re.sub(r"^\s{0,3}#{1,6}\s+", "", title)
        title = re.sub(r"^\s*[-*+]\s+", "", title)
        title = re.sub(r"\s+[-|·]\s+.*$", "", title)
        return normalize_whitespace(title)
