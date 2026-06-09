from __future__ import annotations

import time

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import IngestRequest
from app.pipeline.orchestrator import KnowledgePipeline
from app.services.embedding_service import EmbeddingService
from app.services.job_service import JobService
from app.services.vector_store import VectorStore


def build_job_service(tmp_path):
    settings = Settings(
        sqlite_path=str(tmp_path / "knowledge.db"),
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        openai_api_key="",
    )
    repo = KnowledgeRepository(settings.sqlite_path)
    vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, EmbeddingService(settings))
    pipeline = KnowledgePipeline(settings, repo, vector_store)
    service = JobService(repo, pipeline)
    return repo, service


def wait_for_terminal(service: JobService, job_id: str) -> dict:
    deadline = time.time() + 10
    while time.time() < deadline:
        job = service.get_job(job_id)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_job_service_runs_ingest_in_background(tmp_path):
    repo, service = build_job_service(tmp_path)
    try:
        job = service.submit_ingest(
            IngestRequest(
                source_type="text",
                source="技术：LangGraph\nLangGraph 用于异步入库任务测试。",
                title="异步任务测试",
            ),
            idempotency_key="ingest-langgraph",
        )

        finished = wait_for_terminal(service, job["id"])
        duplicate = service.submit_ingest(
            IngestRequest(source_type="text", source="不会重复执行", title="重复任务"),
            idempotency_key="ingest-langgraph",
        )

        assert finished["status"] == "succeeded"
        assert finished["result"]["document_id"]
        assert repo.get_document(finished["result"]["document_id"])
        assert duplicate["id"] == job["id"]
        assert any(event["event_type"] == "succeeded" for event in finished["events"])
    finally:
        service.shutdown()


def test_job_service_lists_recent_jobs_with_events(tmp_path):
    _, service = build_job_service(tmp_path)
    try:
        job = service.submit_ingest(
            IngestRequest(source_type="text", source="任务列表测试", title="任务列表")
        )
        listed = service.list_jobs(limit=5)

        assert listed[0]["id"] == job["id"]
        assert listed[0]["events"]
        assert listed[0]["events"][0]["event_type"] == "queued"
    finally:
        service.shutdown()


def test_job_service_runs_reindex_in_background(tmp_path):
    repo, service = build_job_service(tmp_path)
    try:
        first = service.submit_ingest(
            IngestRequest(source_type="text", source="LangGraph 多 Agent 编排", title="Job Reindex A")
        )
        second = service.submit_ingest(
            IngestRequest(source_type="text", source="LangGraph 流程节点编排", title="Job Reindex B")
        )
        wait_for_terminal(service, first["id"])
        wait_for_terminal(service, second["id"])

        reindex = service.submit_reindex()
        finished = wait_for_terminal(service, reindex["id"])

        assert finished["status"] == "succeeded"
        assert finished["job_type"] == "reindex"
        assert finished["result"]["documents"] >= 2
        assert finished["result"]["links_rebuilt"] >= 0
        assert len(repo.list_jobs(limit=5)) >= 3
    finally:
        service.shutdown()


def test_job_service_runs_document_enrichment_in_background(tmp_path):
    repo, service = build_job_service(tmp_path)
    try:
        ingest = service.submit_ingest(
            IngestRequest(source_type="text", source="技术：LangGraph\nLangGraph 多 Agent 编排", title="Job Enrich")
        )
        ingested = wait_for_terminal(service, ingest["id"])
        document_id = ingested["result"]["document_id"]
        repo.replace_links(document_id, [])
        repo.replace_document_graph(document_id, [], [])

        enrich = service.submit_enrich(document_id)
        finished = wait_for_terminal(service, enrich["id"])

        assert finished["status"] == "succeeded"
        assert finished["job_type"] == "enrich"
        assert finished["result"]["document_id"] == document_id
        assert repo.list_document_entities(document_id)
    finally:
        service.shutdown()


def test_job_service_retries_failed_job(tmp_path):
    _, service = build_job_service(tmp_path)
    try:
        job = service.submit_ingest(
            IngestRequest(source_type="pdf", source=str(tmp_path / "missing.pdf"), title="缺失文件")
        )
        failed = wait_for_terminal(service, job["id"])
        retried = service.retry(job["id"])

        assert failed["status"] == "failed"
        assert retried["status"] in {"queued", "running", "failed"}
        assert any(event["event_type"] == "queued" for event in retried["events"])
    finally:
        service.shutdown()
