from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from app.db import KnowledgeRepository
from app.models import IngestRequest
from app.pipeline.orchestrator import KnowledgePipeline


class JobService:
    def __init__(self, repo: KnowledgeRepository, pipeline: KnowledgePipeline) -> None:
        self.repo = repo
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-job")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit_ingest(self, request: IngestRequest, idempotency_key: str | None = None) -> dict[str, Any]:
        job = self.repo.create_job(
            job_id=f"job-{uuid.uuid4().hex[:24]}",
            job_type="ingest",
            payload=request.model_dump(),
            idempotency_key=idempotency_key,
        )
        if not self.repo.list_job_events(job["id"]):
            self.repo.add_job_event(job["id"], "queued", "任务已进入队列。")
        if job["status"] == "queued":
            self._schedule(job["id"])
        return self.get_job(job["id"])

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self.repo.get_job(job_id)
        if not job:
            raise ValueError("任务不存在。")
        job["events"] = self.repo.list_job_events(job_id)
        return job

    def cancel(self, job_id: str) -> dict[str, Any]:
        if not self.repo.get_job(job_id):
            raise ValueError("任务不存在。")
        cancelled = self.repo.cancel_job(job_id)
        if not cancelled:
            raise ValueError("只有排队中的任务可以取消。")
        self.repo.add_job_event(job_id, "cancelled", "任务已取消。")
        return self.get_job(job_id)

    def retry(self, job_id: str) -> dict[str, Any]:
        if not self.repo.get_job(job_id):
            raise ValueError("任务不存在。")
        queued = self.repo.retry_job(job_id)
        if not queued:
            raise ValueError("只有失败任务可以重试。")
        self.repo.add_job_event(job_id, "queued", "任务已重新进入队列。")
        self._schedule(job_id)
        return self.get_job(job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _schedule(self, job_id: str) -> None:
        with self._lock:
            existing = self._futures.get(job_id)
            if existing and not existing.done():
                return
            self._futures[job_id] = self.executor.submit(self._run_job, job_id)

    def _run_job(self, job_id: str) -> None:
        if not self.repo.mark_job_running(job_id):
            return
        self.repo.add_job_event(job_id, "running", "任务开始执行。")
        job = self.repo.get_job(job_id)
        if not job:
            return
        try:
            if job["job_type"] != "ingest":
                raise ValueError(f"不支持的任务类型：{job['job_type']}")
            request = IngestRequest(**job["payload"])
            result = self.pipeline.ingest(request)
            self.repo.complete_job(job_id, result)
            self.repo.add_job_event(job_id, "succeeded", "任务执行成功。", {"document_id": result.get("document_id")})
        except Exception as error:
            self.repo.fail_job(job_id, str(error))
            self.repo.add_job_event(job_id, "failed", f"任务执行失败：{error}")
