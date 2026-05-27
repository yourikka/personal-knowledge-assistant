from __future__ import annotations

from app.models import PipelineState
from app.services.parser_utils import parse_content


class ParserAgent:
    def run(self, state: PipelineState) -> PipelineState:
        parsed_text, metadata = parse_content(
            source_type=state.request.source_type,
            raw_bytes=state.raw_bytes,
            metadata=state.metadata,
        )
        state.parsed_text = parsed_text
        state.metadata = metadata
        state.title = state.request.title or metadata.get("detected_title") or "未命名文档"
        state.logs.append(f"parser: 已完成格式解析和元数据提取，提取 {len(parsed_text)} 字符。")
        return state
