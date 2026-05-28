from __future__ import annotations

from app.config import Settings
from app.models import PipelineState
from app.services.chunking import DocumentChunker


class ChunkingAgent:
    def __init__(self, settings: Settings) -> None:
        self.chunker = DocumentChunker(
            target_chars=settings.chunk_target_chars,
            overlap_chars=settings.chunk_overlap_chars,
            min_chars=settings.chunk_min_chars,
            max_chars=settings.chunk_max_chars,
        )

    def run(self, state: PipelineState) -> PipelineState:
        state.chunks = self.chunker.chunk(document_id=state.document_id, text=state.cleaned_text)
        state.sections = self.chunker.sections(document_id=state.document_id, text=state.cleaned_text)
        chunk_lengths = [len(chunk["text"]) for chunk in state.chunks]
        state.metadata["chunking"] = {
            "strategy": "heading_paragraph_sentence_overlap",
            "count": len(state.chunks),
            "section_count": len(state.sections),
            "min_chars": min(chunk_lengths) if chunk_lengths else 0,
            "max_chars": max(chunk_lengths) if chunk_lengths else 0,
            "avg_chars": round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0,
        }
        state.logs.append(
            f"chunking: 已按标题/段落/句子边界切分 {len(state.chunks)} 个 chunk，并生成 {len(state.sections)} 个 section。"
        )
        return state
