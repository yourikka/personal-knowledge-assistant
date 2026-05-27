from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


SourceType = Literal["url", "pdf", "markdown", "image", "text", "html"]


class IngestRequest(BaseModel):
    source_type: SourceType
    source: str = Field(min_length=1, description="URL、本地文件路径或内联内容")
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    session_id: str | None = Field(default=None, min_length=1)


class DocumentResponse(BaseModel):
    id: str
    title: str
    source_type: str
    source_uri: str
    category: str
    confidence: float
    tags: list[str]
    summary: str
    created_at: str
    related: list[dict[str, Any]] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document_id: str
    duplicate: bool
    title: str
    category: str
    tags: list[str]
    summary: str
    related: list[dict[str, Any]]
    logs: list[str]
    graph: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    references: list[dict[str, Any]]
    session_id: str | None
    logs: list[str]


class HealthResponse(BaseModel):
    status: str
    chroma_enabled: bool
    playwright_enabled: bool


class ReindexResponse(BaseModel):
    status: str
    documents: int
    links_rebuilt: int


class ImageGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    size: str = Field(default="1024x1024")
    quality: str = Field(default="high")


class ImageGenerateResponse(BaseModel):
    prompt: str
    revised_prompt: str
    model: str
    image_b64: str | None = None
    image_url: str | None = None
    logs: list[str]


@dataclass(slots=True)
class PipelineState:
    request: IngestRequest
    document_id: str = ""
    fingerprint: str = ""
    source_uri: str = ""
    raw_bytes: bytes = b""
    raw_text: str = ""
    parsed_text: str = ""
    cleaned_text: str = ""
    chunks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    title: str = ""
    category: str = "未分类"
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""
    related: list[dict[str, Any]] = field(default_factory=list)
    duplicate_of: str | None = None
    graph: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
