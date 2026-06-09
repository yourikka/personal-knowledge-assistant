from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import IngestRequest
from app.pipeline.orchestrator import KnowledgePipeline
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def build_pipeline(workdir: Path) -> KnowledgePipeline:
    settings = Settings(
        sqlite_path=str(workdir / "knowledge.db"),
        chroma_dir=str(workdir / "chroma"),
        enable_chroma=False,
        openai_api_key="",
        embedding_provider="local",
        memory_enabled=False,
        query_cache_enabled=False,
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, EmbeddingService(settings))
    return KnowledgePipeline(settings, repo, vector_store)


def timed(label: str, fn: Callable[[], Any]) -> tuple[str, float, Any]:
    start = time.perf_counter()
    result = fn()
    return label, time.perf_counter() - start, result


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * ratio)))
    return sorted(values)[index]


def summarize(samples: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "count": len(values),
            "avg_seconds": round(statistics.fmean(values), 4) if values else 0.0,
            "p50_seconds": round(percentile(values, 0.50), 4),
            "p95_seconds": round(percentile(values, 0.95), 4),
            "max_seconds": round(max(values), 4) if values else 0.0,
        }
        for name, values in samples.items()
    }


def run_benchmark(iterations: int) -> dict[str, Any]:
    samples: dict[str, list[float]] = {
        "ingest_fast": [],
        "enrich_document": [],
        "query_fast": [],
        "query_model_fallback": [],
    }
    document_ids: list[str] = []

    with tempfile.TemporaryDirectory(prefix="pka-benchmark-") as temp_dir:
        pipeline = build_pipeline(Path(temp_dir))
        for index in range(iterations):
            request = IngestRequest(
                source_type="markdown",
                source=(
                    f"# LangGraph Benchmark {index}\n\n"
                    "技术：LangGraph\n"
                    "组织：OpenAI\n"
                    "概念：RAG 编排\n\n"
                    "LangGraph 用于构建多 Agent 工作流，个人知识库会把文档切块、生成本地元数据、"
                    "后台补齐图谱、向量索引和相似文档链接。\n\n"
                    "快速 RAG 模式会跳过模型生成，优先返回带引用的本地摘要。"
                ),
                title=f"LangGraph Benchmark {index}",
            )

            _, duration, result = timed("ingest_fast", lambda request=request: pipeline.ingest_fast(request))
            samples["ingest_fast"].append(duration)
            document_id = result["document_id"]
            document_ids.append(document_id)

            _, duration, _ = timed("enrich_document", lambda document_id=document_id: pipeline.enrich_document(document_id))
            samples["enrich_document"].append(duration)

            _, duration, _ = timed(
                "query_fast",
                lambda index=index: pipeline.query(
                    query=f"LangGraph Benchmark {index} 的快速 RAG 模式做了什么？",
                    top_k=3,
                    session_id=f"bench-fast-{index}",
                    answer_mode="fast",
                ),
            )
            samples["query_fast"].append(duration)

            _, duration, _ = timed(
                "query_model_fallback",
                lambda index=index: pipeline.query(
                    query=f"LangGraph Benchmark {index} 如何补齐图谱和链接？",
                    top_k=3,
                    session_id=f"bench-model-{index}",
                    answer_mode="model",
                ),
            )
            samples["query_model_fallback"].append(duration)

        stats = pipeline.repo.stats()
        vector_stats = pipeline.vector_store.stats()

    return {
        "iterations": iterations,
        "documents": len(document_ids),
        "summary": summarize(samples),
        "repository": stats,
        "vector_store": vector_stats,
        "notes": [
            "benchmark uses local embeddings and disables external model calls by default",
            "query_model_fallback measures model-mode overhead when no OpenAI key is configured",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local performance baseline for ingest, enrich and RAG query.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of documents and query rounds.")
    parser.add_argument("--output", help="Optional path to write JSON result.")
    args = parser.parse_args()

    result = run_benchmark(max(1, args.iterations))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
