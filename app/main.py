from __future__ import annotations

from contextlib import asynccontextmanager
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import get_settings
from .db import KnowledgeRepository
from .models import (
    DocumentResponse,
    HealthResponse,
    ImageGenerateRequest,
    ImageGenerateResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
)
from .pipeline.orchestrator import KnowledgePipeline
from .services.vector_store import VectorStore


settings = get_settings()
repo = KnowledgeRepository(settings.sqlite_path)
vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma)
pipeline = KnowledgePipeline(settings, repo, vector_store)


@asynccontextmanager
async def lifespan(_: FastAPI):
    pipeline.bootstrap()
    yield


app = FastAPI(title="Personal Knowledge Assistant", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chroma_enabled=vector_store.collection is not None,
        playwright_enabled=settings.enable_playwright,
    )


@app.post("/api/knowledge/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest) -> IngestResponse:
    try:
        result = pipeline.ingest(request)
        return IngestResponse(**result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"入库失败：{error}") from error


@app.post("/api/knowledge/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    source_type: str = Form(...),
    title: str | None = Form(default=None),
) -> IngestResponse:
    try:
        suffix = os.path.splitext(file.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        try:
            request = IngestRequest(
                source_type=source_type,
                source=temp_path,
                title=title or file.filename,
                metadata={"display_source_uri": f"upload://{file.filename or os.path.basename(temp_path)}"},
            )
            result = pipeline.ingest(request)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return IngestResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"上传入库失败：{error}") from error


@app.post("/api/knowledge/query", response_model=QueryResponse)
def query_knowledge(request: QueryRequest) -> QueryResponse:
    try:
        result = pipeline.query(query=request.query, top_k=request.top_k, session_id=request.session_id)
        return QueryResponse(session_id=request.session_id, **result)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"检索失败：{error}") from error


@app.get("/api/knowledge/documents", response_model=list[DocumentResponse])
def list_documents(limit: int = 20) -> list[DocumentResponse]:
    documents = repo.list_documents(limit=limit)
    return [DocumentResponse(**document, related=repo.list_links(document["id"])) for document in documents]


@app.get("/api/knowledge/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str) -> DocumentResponse:
    document = repo.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在。")
    return DocumentResponse(**document, related=repo.list_links(document_id))


@app.post("/api/knowledge/reindex", response_model=ReindexResponse)
def rebuild_index_and_links() -> ReindexResponse:
    try:
        result = pipeline.rebuild_links()
        return ReindexResponse(**result)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"重建索引失败：{error}") from error


@app.post("/api/images/generate", response_model=ImageGenerateResponse)
def generate_image(request: ImageGenerateRequest) -> ImageGenerateResponse:
    try:
        result = pipeline.generate_image(prompt=request.prompt, size=request.size, quality=request.quality)
        return ImageGenerateResponse(**result)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"生图失败：{error}") from error
