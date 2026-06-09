from __future__ import annotations

import pytest

from app.models import IngestRequest, PipelineState
from app.pipeline.agents.parser_agent import ParserAgent
from app.services.parser_utils import parse_content, restore_document_structure


def test_restore_document_structure_detects_headings_and_tables():
    text = """1. Overview
LangGraph 用于多 Agent 编排，支持状态流转和节点连接。

Name    Role    Score
LangGraph    Workflow    0.95
"""

    restored, headings, table_count = restore_document_structure(text)

    assert "# 1. Overview" in restored
    assert "| Name | Role | Score |" in restored
    assert headings == ["1. Overview"]
    assert table_count == 2


def test_pdf_parser_fallback_keeps_structure_metadata():
    raw = b"1. Overview\nLangGraph workflow text.\n\nName    Role\nAgent    Worker"

    text, metadata = parse_content("pdf", raw, {"filename": "note.pdf"})

    assert "## Page 1" in text
    assert "| Name | Role |" in text
    assert metadata["page_count"] == 1
    assert metadata["structure"]["table_count"] >= 1
    assert metadata["structure"]["headings"]


def test_parser_agent_cleans_detected_title_and_rejects_empty_text():
    agent = ParserAgent()
    state = PipelineState(
        request=IngestRequest(source_type="markdown", source="#   LangGraph   标题\n\n正文", title=None),
        raw_bytes=b"#   LangGraph   \xe6\xa0\x87\xe9\xa2\x98\n\n\xe6\xad\xa3\xe6\x96\x87",
        source_uri="inline://content",
    )

    parsed = agent.run(state)

    assert parsed.title == "LangGraph 标题"

    empty_state = PipelineState(
        request=IngestRequest(source_type="text", source="   "),
        raw_bytes=b"   ",
        source_uri="inline://content",
    )
    with pytest.raises(ValueError, match="解析结果为空"):
        agent.run(empty_state)
