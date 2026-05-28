from __future__ import annotations

from pathlib import Path

from scripts.eval_rag import run_eval


def test_rag_eval_runner_returns_metrics():
    result = run_eval(Path("evals/rag_cases.jsonl"), top_k=2)

    assert result["queries"] >= 1
    assert 0 <= result["doc_recall"] <= 1
    assert 0 <= result["answer_term_coverage"] <= 1
    assert result["details"]
