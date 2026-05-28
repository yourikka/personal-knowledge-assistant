from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import IngestRequest
from app.pipeline.orchestrator import KnowledgePipeline
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid jsonl at {path}:{line_number}: {error}") from error
    return cases


def build_pipeline(workdir: Path) -> KnowledgePipeline:
    settings = Settings(
        sqlite_path=str(workdir / "knowledge.db"),
        chroma_dir=str(workdir / "chroma"),
        enable_chroma=False,
        openai_api_key="",
        embedding_provider="local",
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, embedding_service)
    return KnowledgePipeline(settings, repo, vector_store)


def run_eval(cases_path: Path, top_k: int = 3) -> dict[str, Any]:
    cases = load_cases(cases_path)
    with tempfile.TemporaryDirectory(prefix="pka-rag-eval-") as temp_dir:
        pipeline = build_pipeline(Path(temp_dir))
        details = []
        total_queries = 0
        doc_hits = 0
        term_hits = 0
        total_references = 0

        for case in cases:
            for document in case.get("documents", []):
                pipeline.ingest(IngestRequest(**document))

            for query_case in case.get("queries", []):
                total_queries += 1
                result = pipeline.query(
                    query=query_case["query"],
                    top_k=top_k,
                    session_id=f"eval-{case['name']}",
                )
                references = result.get("references", [])
                answer = result.get("answer", "")
                reference_titles = {item.get("title", "") for item in references}
                expected_titles = set(query_case.get("expected_doc_titles", []))
                expected_terms = [str(item) for item in query_case.get("expected_terms", [])]
                matched_terms = [term for term in expected_terms if term in answer]
                doc_hit = bool(expected_titles & reference_titles) if expected_titles else bool(references)
                term_hit = len(matched_terms) == len(expected_terms) if expected_terms else True
                doc_hits += int(doc_hit)
                term_hits += int(term_hit)
                total_references += len(references)
                details.append(
                    {
                        "case": case["name"],
                        "query": query_case["query"],
                        "doc_hit": doc_hit,
                        "term_hit": term_hit,
                        "matched_terms": matched_terms,
                        "expected_terms": expected_terms,
                        "reference_titles": sorted(reference_titles),
                        "reference_count": len(references),
                    }
                )

    return {
        "cases": len(cases),
        "queries": total_queries,
        "doc_recall": round(doc_hits / max(total_queries, 1), 4),
        "answer_term_coverage": round(term_hits / max(total_queries, 1), 4),
        "avg_references": round(total_references / max(total_queries, 1), 2),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation cases.")
    parser.add_argument("--cases", default="evals/rag_cases.jsonl", help="Path to jsonl eval cases.")
    parser.add_argument("--top-k", type=int, default=3, help="Query top_k.")
    parser.add_argument("--output", help="Optional path to write JSON result.")
    args = parser.parse_args()

    result = run_eval(Path(args.cases), top_k=args.top_k)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
