from __future__ import annotations

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
