from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .db import KnowledgeRepository
from .models import (
    AsyncIngestRequest,
    DeleteDocumentResponse,
    DeleteMemoryResponse,
    DocumentChunkResponse,
    DocumentDetailResponse,
    DocumentGraphResponse,
    DocumentResponse,
    HealthResponse,
    ImageGenerateRequest,
    ImageGenerateResponse,
    IngestRequest,
    IngestResponse,
    JobResponse,
    MemoryResponse,
    MemoryUpdateRequest,
    DocumentClickRequest,
    PersonalizationEventResponse,
    QueryRequest,
    QueryFeedbackRequest,
    QueryResponse,
    ReindexDocumentResponse,
    ReindexResponse,
)
from .pipeline.orchestrator import KnowledgePipeline
from .services.embedding_service import EmbeddingService
from .services.job_service import JobService
from .services.vector_store import VectorStore


settings = get_settings()
repo = KnowledgeRepository(settings.sqlite_path)
embedding_service = EmbeddingService(settings)
vector_store = VectorStore(settings.chroma_dir, settings.enable_chroma, embedding_service)
pipeline = KnowledgePipeline(settings, repo, vector_store)
job_service = JobService(repo, pipeline)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    pipeline.bootstrap()
    try:
        yield
    finally:
        job_service.shutdown()


app = FastAPI(title="Personal Knowledge Assistant", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def prevent_frontend_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chroma_enabled=vector_store.collection is not None,
        playwright_enabled=settings.enable_playwright,
        repository=repo.stats(),
        vector_store=vector_store.stats(),
    )


@app.post("/api/knowledge/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest) -> IngestResponse:
    try:
        result = pipeline.ingest_fast(request)
        if not result["duplicate"]:
            job_service.submit_enrich(result["document_id"], idempotency_key=f"enrich-{result['document_id']}")
            result["logs"].append("jobs: 已提交后台文档增强任务。")
        return IngestResponse(**result)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"入库失败：{error}") from error


@app.post("/api/jobs/ingest", response_model=JobResponse)
def submit_ingest_job(request: AsyncIngestRequest) -> JobResponse:
    try:
        return JobResponse(**job_service.submit_ingest(request.request, request.idempotency_key))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"提交任务失败：{error}") from error


@app.post("/api/jobs/reindex", response_model=JobResponse)
def submit_reindex_job() -> JobResponse:
    try:
        return JobResponse(**job_service.submit_reindex(idempotency_key=None))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"提交重建任务失败：{error}") from error


@app.post("/api/jobs/documents/{document_id}/enrich", response_model=JobResponse)
def submit_document_enrich_job(document_id: str) -> JobResponse:
    if not repo.get_document(document_id):
        raise HTTPException(status_code=404, detail="文档不存在。")
    try:
        return JobResponse(**job_service.submit_enrich(document_id, idempotency_key=f"enrich-{document_id}"))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"提交文档增强任务失败：{error}") from error


@app.get("/api/jobs", response_model=list[JobResponse])
def list_jobs(limit: int = 20) -> list[JobResponse]:
    return [JobResponse(**job) for job in job_service.list_jobs(limit=limit)]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    try:
        return JobResponse(**job_service.get_job(job_id))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str) -> JobResponse:
    try:
        return JobResponse(**job_service.cancel(job_id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/jobs/{job_id}/retry", response_model=JobResponse)
def retry_job(job_id: str) -> JobResponse:
    try:
        return JobResponse(**job_service.retry(job_id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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


@app.post("/api/knowledge/query/stream")
def stream_query_knowledge(request: QueryRequest) -> StreamingResponse:
    def events():
        try:
            for item in pipeline.query_stream(
                query=request.query,
                top_k=request.top_k,
                session_id=request.session_id,
            ):
                payload = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {item['event']}\ndata: {payload}\n\n"
        except Exception as error:
            payload = json.dumps({"error": f"流式检索失败：{error}"}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/personalization/clicks", response_model=PersonalizationEventResponse)
def record_document_click(request: DocumentClickRequest) -> PersonalizationEventResponse:
    if not repo.get_document(request.document_id):
        raise HTTPException(status_code=404, detail="文档不存在。")
    pipeline.personalization_service.record_click(
        session_id=request.session_id,
        document_id=request.document_id,
        query=request.query,
    )
    return PersonalizationEventResponse(status="ok")


@app.post("/api/personalization/feedback", response_model=PersonalizationEventResponse)
def record_query_feedback(request: QueryFeedbackRequest) -> PersonalizationEventResponse:
    if request.document_id and not repo.get_document(request.document_id):
        raise HTTPException(status_code=404, detail="文档不存在。")
    pipeline.personalization_service.record_feedback(
        session_id=request.session_id,
        query=request.query,
        rating=request.rating,
        document_id=request.document_id,
        comment=request.comment,
    )
    return PersonalizationEventResponse(status="ok")


@app.get("/api/memories", response_model=list[MemoryResponse])
def list_memories(session_id: str | None = None, limit: int = 20) -> list[MemoryResponse]:
    memories = repo.list_memories(session_id=session_id, limit=limit)
    return [MemoryResponse(**memory) for memory in memories]


@app.patch("/api/memories/{memory_id}", response_model=MemoryResponse)
def update_memory(memory_id: str, request: MemoryUpdateRequest) -> MemoryResponse:
    content = None
    if request.content is not None:
        content = request.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="记忆内容不能为空。")
    tags = None
    if request.tags is not None:
        tags = [tag.strip() for tag in request.tags if tag.strip()][:5]
    updated = repo.update_memory(
        memory_id,
        content=content,
        importance=request.importance,
        tags=tags,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="记忆不存在。")
    vector_store.add_memory(
        memory_id=updated["id"],
        text=updated["content"],
        metadata={
            "session_id": updated.get("session_id") or "",
            "kind": "memory",
            "scope": updated.get("scope", "session"),
            "memory_kind": updated["kind"],
            "importance": float(updated.get("importance", 0.5)),
        },
    )
    return MemoryResponse(**updated)


@app.delete("/api/memories/{memory_id}", response_model=DeleteMemoryResponse)
def delete_memory(memory_id: str) -> DeleteMemoryResponse:
    if not repo.delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="记忆不存在。")
    vector_store.delete_ids([memory_id])
    return DeleteMemoryResponse(status="ok", memory_id=memory_id)


@app.get("/api/knowledge/documents", response_model=list[DocumentResponse])
def list_documents(limit: int = 20) -> list[DocumentResponse]:
    documents = repo.list_documents(limit=limit)
    return [DocumentResponse(**document, related=repo.list_links(document["id"])) for document in documents]


@app.get("/api/knowledge/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str) -> DocumentDetailResponse:
    document = repo.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在。")
    return DocumentDetailResponse(**document, related=repo.list_links(document_id))


@app.get("/api/knowledge/documents/{document_id}/chunks", response_model=list[DocumentChunkResponse])
def list_document_chunks(document_id: str) -> list[DocumentChunkResponse]:
    if not repo.get_document(document_id):
        raise HTTPException(status_code=404, detail="文档不存在。")
    return [DocumentChunkResponse(**chunk) for chunk in repo.list_document_chunks(document_id)]


@app.get("/api/knowledge/documents/{document_id}/graph", response_model=DocumentGraphResponse)
def get_document_graph(document_id: str) -> DocumentGraphResponse:
    if not repo.get_document(document_id):
        raise HTTPException(status_code=404, detail="文档不存在。")
    graph = pipeline.graph_service.graph_view(document_id)
    return DocumentGraphResponse(document_id=document_id, **graph)


def delete_document_response(document_id: str) -> DeleteDocumentResponse:
    try:
        result = pipeline.delete_document(document_id)
        return DeleteDocumentResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"删除文档失败：{error}") from error


@app.delete("/api/knowledge/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str) -> DeleteDocumentResponse:
    return delete_document_response(document_id)


@app.post("/api/knowledge/documents/{document_id}/delete", response_model=DeleteDocumentResponse)
def delete_document_via_post(document_id: str) -> DeleteDocumentResponse:
    return delete_document_response(document_id)


@app.post("/api/knowledge/reindex", response_model=ReindexResponse)
def rebuild_index_and_links() -> ReindexResponse:
    try:
        result = pipeline.rebuild_links()
        return ReindexResponse(**result)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"重建索引失败：{error}") from error


@app.post("/api/knowledge/documents/{document_id}/reindex", response_model=ReindexDocumentResponse)
def reindex_document(document_id: str) -> ReindexDocumentResponse:
    try:
        result = pipeline.reindex_document(document_id)
        return ReindexDocumentResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"重建文档索引失败：{error}") from error


@app.post("/api/images/generate", response_model=ImageGenerateResponse)
def generate_image(request: ImageGenerateRequest) -> ImageGenerateResponse:
    try:
        result = pipeline.generate_image(prompt=request.prompt, size=request.size, quality=request.quality)
        return ImageGenerateResponse(**result)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"生图失败：{error}") from error
