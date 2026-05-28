from __future__ import annotations

from app.config import Settings
from app.services.self_check_service import SelfCheckService


def test_self_check_repairs_summary_and_classification(tmp_path):
    service = SelfCheckService(Settings(sqlite_path=str(tmp_path / "knowledge.db")))
    source = "LangGraph 用于构建多 Agent 工作流。" * 12

    summary, summary_logs = service.check_summary("太短", source)
    category, confidence, tags, class_logs = service.check_classification(
        category="未知",
        confidence=0.2,
        tags=["LangGraph"],
        source_text=source,
    )

    assert len(summary) > len("太短")
    assert summary_logs
    assert category == "学习"
    assert confidence >= 0.7
    assert len(tags) >= 3
    assert class_logs


def test_self_check_repairs_answer_citations(tmp_path):
    service = SelfCheckService(Settings(sqlite_path=str(tmp_path / "knowledge.db")))

    answer, logs = service.check_answer(
        "LangGraph 适合做 Agent 编排。[9]",
        references=[{"id": "doc-1"}, {"id": "doc-2"}],
    )

    assert "[9]" not in answer
    assert "[1]" in answer
    assert logs
