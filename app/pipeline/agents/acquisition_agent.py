from __future__ import annotations

import hashlib
import uuid

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import PipelineState
from app.services.parser_utils import read_source_payload


class AcquisitionAgent:
    def __init__(self, settings: Settings, repo: KnowledgeRepository) -> None:
        self.settings = settings
        self.repo = repo

    def run(self, state: PipelineState) -> PipelineState:
        raw_bytes, source_uri, metadata = read_source_payload(
            source_type=state.request.source_type,
            source=state.request.source,
            enable_playwright=self.settings.enable_playwright,
            blacklist=self.settings.url_blacklist,
        )
        if len(raw_bytes) > self.settings.max_source_bytes:
            raise ValueError(f"数据源过大：{len(raw_bytes)} bytes，超过 MAX_SOURCE_BYTES。")
        fingerprint = hashlib.sha256(
            f"{state.request.source_type}:{source_uri}:".encode("utf-8") + raw_bytes
        ).hexdigest()
        existing = self.repo.get_document_by_fingerprint(fingerprint)

        state.document_id = existing["id"] if existing else uuid.uuid4().hex
        state.fingerprint = fingerprint
        state.source_uri = state.request.metadata.get("display_source_uri", source_uri)
        state.raw_bytes = raw_bytes
        state.metadata = {
            **metadata,
            **state.request.metadata,
            "fingerprint": fingerprint,
            "source_type": state.request.source_type,
            "source_uri": state.source_uri,
            "raw_bytes": len(raw_bytes),
        }
        state.duplicate_of = existing["id"] if existing else None
        state.logs.append(
            "acquisition: 已完成数据采集与去重检查。"
            + (f" 命中重复文档 {existing['id']}。" if existing else f" 读取 {len(raw_bytes)} bytes。")
        )
        return state
