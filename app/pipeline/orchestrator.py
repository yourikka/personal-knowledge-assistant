from __future__ import annotations

from app.config import Settings
from app.db import KnowledgeRepository
from app.models import IngestRequest, PipelineState
from app.pipeline.agents.acquisition_agent import AcquisitionAgent
from app.pipeline.agents.cleaning_agent import CleaningAgent
from app.pipeline.agents.chunking_agent import ChunkingAgent
from app.pipeline.agents.classification_agent import ClassificationAgent
from app.pipeline.agents.linking_agent import LinkingAgent
from app.pipeline.agents.parser_agent import ParserAgent
from app.pipeline.agents.query_agent import QueryAgent
from app.pipeline.agents.summary_agent import SummaryAgent
from app.pipeline.agents.image_generation_agent import ImageGenerationAgent
from app.services.memory_service import MemoryService
from app.services.openai_client import OpenAIService
from app.services.rag_service import RAGService
from app.services.chunking import DocumentChunker
from app.services.graph_service import GraphExtractionService
from app.services.personalization_service import PersonalizationService
from app.services.query_cache import QueryCacheService
from app.services.self_check_service import SelfCheckService
from app.services.vector_store import VectorStore

from langgraph.graph import END, StateGraph


class KnowledgePipeline:
    def __init__(self, settings: Settings, repo: KnowledgeRepository, vector_store: VectorStore) -> None:
        self.repo = repo
        self.vector_store = vector_store
        self._bootstrapped = False
        self.openai_service = OpenAIService(settings)
        self.acquisition = AcquisitionAgent(settings, repo)
        self.parser = ParserAgent()
        self.cleaning = CleaningAgent()
        self.chunking = ChunkingAgent(settings)
        self.chunker = DocumentChunker(
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
        )
        self.self_check = SelfCheckService(settings)
        self.personalization_service = PersonalizationService(settings, repo)
        self.query_cache = QueryCacheService(settings)
        self.classification = ClassificationAgent(self.openai_service, self.self_check)
        self.summary = SummaryAgent(self.openai_service, self.self_check)
        self.graph_service = GraphExtractionService(settings, repo)
        self.rag_service = RAGService(
            settings,
            repo,
            vector_store,
            self.openai_service,
            self.graph_service,
            self.personalization_service,
            self.query_cache,
        )
        self.memory_service = MemoryService(settings, repo, vector_store, self.openai_service)
        self.linking = LinkingAgent(settings, repo, vector_store, self.rag_service)
        self.query_agent = QueryAgent(
            settings,
            repo,
            vector_store,
            self.openai_service,
            self.rag_service,
            self.memory_service,
            self.self_check,
            self.personalization_service,
        )
        self.image_generation_agent = ImageGenerationAgent(settings, self.openai_service)
        self.ingest_graph = self._build_ingest_graph()
        self.bootstrap()

    def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        self.vector_store.reset()
        for document in self.repo.iter_documents():
            chunks = self.repo.list_document_chunks(document["id"])
            if not chunks:
                chunks = self.chunker.chunk(document_id=document["id"], text=document["cleaned_text"])
                self.repo.replace_document_chunks(document["id"], chunks)
            sections = self.repo.list_document_sections(document["id"])
            if not sections:
                sections = self.chunker.sections(document_id=document["id"], text=document["cleaned_text"])
                self.repo.replace_document_sections(document["id"], sections)

            self.vector_store.add_document(
                document_id=document["id"],
                text=document["cleaned_text"],
                metadata={"title": document["title"], "category": document["category"]},
            )
            for section in sections:
                self.vector_store.add_section(
                    section_id=section["id"],
                    text=section["text"],
                    metadata={
                        "document_id": document["id"],
                        "section_index": section["section_index"],
                        "heading": section["heading"],
                        "title": document["title"],
                        "category": document["category"],
                    },
                )
            for chunk in chunks:
                self.vector_store.add_chunk(
                    chunk_id=chunk["id"],
                    text=chunk["text"],
                    metadata={
                        "document_id": document["id"],
                        "chunk_index": chunk["chunk_index"],
                        "title": document["title"],
                        "category": document["category"],
                    },
                )
        self.memory_service.bootstrap()
        self._bootstrapped = True

    def ingest(self, request: IngestRequest) -> dict:
        state = self._coerce_state(self.ingest_graph.invoke(PipelineState(request=request)))
        if state.duplicate_of:
            document = self.repo.get_document(state.duplicate_of)
            return {
                "document_id": document["id"],
                "duplicate": True,
                "title": document["title"],
                "category": document["category"],
                "tags": document["tags"],
                "summary": document["summary"],
                "related": self.repo.list_links(document["id"]),
                "graph": self._document_graph_response(document),
                "logs": state.logs + ["orchestrator: 命中去重，直接返回已有文档。"],
            }
        return {
            "document_id": state.document_id,
            "duplicate": False,
            "title": state.title,
            "category": state.category,
            "tags": state.tags,
            "summary": state.summary,
            "related": state.related,
            "graph": state.graph,
            "logs": state.logs + ["orchestrator: LangGraph 入库图执行完成。"],
        }

    def query(self, query: str, top_k: int, session_id: str | None = None) -> dict:
        return self.query_agent.run(query=query, top_k=top_k, session_id=session_id)

    def query_stream(self, query: str, top_k: int, session_id: str | None = None):
        result = self.query(query=query, top_k=top_k, session_id=session_id)
        chunk_size = max(12, self.query_agent.settings.query_stream_chunk_chars)
        answer = result["answer"]
        for index in range(0, len(answer), chunk_size):
            yield {"event": "delta", "data": answer[index : index + chunk_size]}
        yield {"event": "references", "data": result["references"]}
        yield {"event": "memories", "data": result["memories"]}
        yield {"event": "logs", "data": result["logs"]}
        yield {"event": "done", "data": {"session_id": session_id}}

    def rebuild_links(self) -> dict:
        self.bootstrap()
        rebuilt = 0
        for document in self.repo.iter_documents():
            related = self._related_links_for_document(document)
            self.repo.replace_links(document["id"], related)
            rebuilt += len(related)
        self.query_cache.clear()
        return {"status": "ok", "documents": len(self.repo.iter_documents()), "links_rebuilt": rebuilt}

    def reindex_document(self, document_id: str) -> dict:
        document = self.repo.get_document(document_id)
        if not document:
            raise ValueError("文档不存在。")

        old_chunk_ids = [chunk["id"] for chunk in self.repo.list_document_chunks(document_id)]
        old_section_ids = [section["id"] for section in self.repo.list_document_sections(document_id)]
        chunks = self.chunker.chunk(document_id=document_id, text=document["cleaned_text"])
        sections = self.chunker.sections_from_chunks(document_id=document_id, chunks=chunks)

        self.vector_store.delete_ids([document_id, *old_section_ids, *old_chunk_ids])
        self.repo.replace_document_chunks(document_id, chunks)
        self.repo.replace_document_sections(document_id, sections)

        self.vector_store.add_document(
            document_id=document_id,
            text=document["cleaned_text"],
            metadata={"title": document["title"], "category": document["category"]},
        )
        for section in sections:
            self.vector_store.add_section(
                section_id=section["id"],
                text=section["text"],
                metadata={
                    "document_id": document_id,
                    "section_index": section["section_index"],
                    "heading": section["heading"],
                    "title": document["title"],
                    "category": document["category"],
                },
            )
        for chunk in chunks:
            self.vector_store.add_chunk(
                chunk_id=chunk["id"],
                text=chunk["text"],
                metadata={
                    "document_id": document_id,
                    "chunk_index": chunk["chunk_index"],
                    "title": document["title"],
                    "category": document["category"],
                },
            )

        graph = self.graph_service.build_for_document(document=document, chunks=chunks)
        related = self._related_links_for_document(document)
        self.repo.replace_links(document_id, related)
        self.query_cache.clear()
        return {
            "status": "ok",
            "document_id": document_id,
            "chunks": len(chunks),
            "sections": len(sections),
            "links_rebuilt": len(related),
            "graph_nodes": len(graph["entities"]),
            "graph_edges": len(graph["edges"]),
        }

    def delete_document(self, document_id: str) -> dict:
        document = self.repo.get_document(document_id)
        if not document:
            raise ValueError("文档不存在。")

        chunk_ids = [chunk["id"] for chunk in self.repo.list_document_chunks(document_id)]
        section_ids = [section["id"] for section in self.repo.list_document_sections(document_id)]
        deleted = self.repo.delete_document(document_id)
        if not deleted:
            raise ValueError("文档不存在。")

        self.vector_store.delete_ids([document_id, *section_ids, *chunk_ids])
        self.query_cache.clear()
        return {"status": "ok", "document_id": document_id, "deleted_chunk_ids": chunk_ids}

    def generate_image(self, prompt: str, size: str, quality: str) -> dict:
        return self.image_generation_agent.run(prompt=prompt, size=size, quality=quality)

    def _coerce_state(self, value) -> PipelineState:
        if isinstance(value, PipelineState):
            return value
        if isinstance(value, dict):
            return PipelineState(**value)
        return value

    def _build_ingest_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("agent_acquisition", self.acquisition.run)
        graph.add_node("agent_parser", self.parser.run)
        graph.add_node("agent_cleaning", self.cleaning.run)
        graph.add_node("agent_chunking", self.chunking.run)
        graph.add_node("agent_classification", self.classification.run)
        graph.add_node("agent_summary", self.summary.run)
        graph.add_node("persist", self._persist_document)
        graph.add_node("agent_graph", self._extract_graph)
        graph.add_node("agent_linking", self.linking.run)

        graph.set_entry_point("agent_acquisition")
        graph.add_conditional_edges(
            "agent_acquisition",
            self._route_after_acquisition,
            {
                "duplicate": END,
                "parse": "agent_parser",
            },
        )
        graph.add_edge("agent_parser", "agent_cleaning")
        graph.add_edge("agent_cleaning", "agent_chunking")
        graph.add_edge("agent_chunking", "agent_classification")
        graph.add_edge("agent_classification", "agent_summary")
        graph.add_edge("agent_summary", "persist")
        graph.add_edge("persist", "agent_graph")
        graph.add_edge("agent_graph", "agent_linking")
        graph.add_edge("agent_linking", END)
        return graph.compile()

    def _route_after_acquisition(self, state: PipelineState) -> str:
        if state.duplicate_of:
            return "duplicate"
        return "parse"

    def _persist_document(self, state: PipelineState) -> PipelineState:
        self.repo.upsert_document(
            {
                "id": state.document_id,
                "fingerprint": state.fingerprint,
                "source_type": state.request.source_type,
                "source_uri": state.source_uri,
                "title": state.title,
                "raw_text": state.parsed_text,
                "cleaned_text": state.cleaned_text,
                "summary": state.summary,
                "category": state.category,
                "confidence": state.confidence,
                "tags": state.tags,
                "metadata": state.metadata,
            }
        )
        self.repo.replace_document_chunks(state.document_id, state.chunks)
        self.repo.replace_document_sections(state.document_id, state.sections)
        self.query_cache.clear()
        state.logs.append(
            f"persist: 已写入 SQLite 元数据仓库、{len(state.sections)} 个 section 和 {len(state.chunks)} 个 chunk。"
        )
        return state

    def _related_links_for_document(self, document: dict) -> list[dict]:
        retrieval = self.rag_service.retrieve(
            query=f"{document['title']}\n{document['summary']}\n{' '.join(document['tags'])}\n{document['cleaned_text'][:2000]}",
            top_k=self.linking.settings.related_top_k,
            exclude_ids={document["id"]},
        )
        grouped: dict[str, dict] = {}
        for item in retrieval["references"]:
            current = grouped.get(item["id"])
            if current is None or item["score"] > current["score"]:
                grouped[item["id"]] = item
        related = []
        for item in sorted(grouped.values(), key=lambda value: value["score"], reverse=True):
            if item["score"] < self.linking.settings.related_score_threshold:
                continue
            related.append({"target_id": item["id"], "score": item["score"]})
        return related

    def _extract_graph(self, state: PipelineState) -> PipelineState:
        document = self.repo.get_document(state.document_id)
        if not document:
            return state
        result = self.graph_service.build_for_document(document=document, chunks=state.chunks)
        state.graph = self.graph_service.graph_view(state.document_id)
        state.logs.append(
            f"graph: 已抽取 {len(result['entities'])} 个实体和 {len(result['edges'])} 条关系。"
        )
        return state

    def _document_graph_response(self, document: dict) -> dict:
        graph = self.graph_service.graph_view(document["id"])
        graph["nodes"].insert(
            0,
            {"id": document["id"], "title": document["title"], "category": document["category"], "type": "document"},
        )
        return graph
