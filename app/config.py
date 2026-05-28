from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8010"))
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/knowledge.db")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    enable_chroma: bool = os.getenv("ENABLE_CHROMA", "false").lower() in {"1", "true", "yes", "on"}
    enable_playwright: bool = os.getenv("ENABLE_PLAYWRIGHT", "false").lower() in {"1", "true", "yes", "on"}
    max_source_bytes: int = int(os.getenv("MAX_SOURCE_BYTES", str(10 * 1024 * 1024)))
    related_top_k: int = int(os.getenv("RELATED_TOP_K", "5"))
    related_score_threshold: float = float(os.getenv("RELATED_SCORE_THRESHOLD", "0.28"))
    default_query_top_k: int = int(os.getenv("DEFAULT_QUERY_TOP_K", "3"))
    rag_rewrite_enabled: bool = os.getenv("RAG_REWRITE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    rag_multi_query_limit: int = int(os.getenv("RAG_MULTI_QUERY_LIMIT", "4"))
    rag_candidate_multiplier: int = int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "5"))
    rag_mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.72"))
    rag_context_char_budget: int = int(os.getenv("RAG_CONTEXT_CHAR_BUDGET", "6000"))
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.01"))
    rag_recent_boost: float = float(os.getenv("RAG_RECENT_BOOST", "0.04"))
    rag_tag_boost: float = float(os.getenv("RAG_TAG_BOOST", "0.06"))
    rag_hierarchical_enabled: bool = os.getenv("RAG_HIERARCHICAL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    rag_document_top_k: int = int(os.getenv("RAG_DOCUMENT_TOP_K", "4"))
    rag_section_top_k: int = int(os.getenv("RAG_SECTION_TOP_K", "8"))
    rag_section_chunk_limit: int = int(os.getenv("RAG_SECTION_CHUNK_LIMIT", "4"))
    graph_enabled: bool = os.getenv("GRAPH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    graph_query_top_k: int = int(os.getenv("GRAPH_QUERY_TOP_K", "6"))
    graph_min_entity_length: int = int(os.getenv("GRAPH_MIN_ENTITY_LENGTH", "2"))
    self_check_enabled: bool = os.getenv("SELF_CHECK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    memory_enabled: bool = os.getenv("MEMORY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    memory_top_k: int = int(os.getenv("MEMORY_TOP_K", "5"))
    memory_min_score: float = float(os.getenv("MEMORY_MIN_SCORE", "0.12"))
    memory_write_limit: int = int(os.getenv("MEMORY_WRITE_LIMIT", "3"))
    memory_max_content_chars: int = int(os.getenv("MEMORY_MAX_CONTENT_CHARS", "500"))
    memory_bootstrap_limit: int = int(os.getenv("MEMORY_BOOTSTRAP_LIMIT", "1000"))
    chunk_target_chars: int = int(os.getenv("CHUNK_TARGET_CHARS", "900"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "160"))
    chunk_min_chars: int = int(os.getenv("CHUNK_MIN_CHARS", "180"))
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "1400"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "").strip()
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embedding-3").strip()
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
    embedding_timeout_seconds: int = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))
    embedding_path: str = os.getenv("EMBEDDING_PATH", "/embeddings").strip()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    openai_text_model: str = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.4").strip()
    openai_image_model: str = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip()
    openai_text_timeout_seconds: int = int(os.getenv("OPENAI_TEXT_TIMEOUT_SECONDS", "60"))
    openai_image_timeout_seconds: int = int(os.getenv("OPENAI_IMAGE_TIMEOUT_SECONDS", "180"))
    openai_chat_completions_path: str = os.getenv("OPENAI_CHAT_COMPLETIONS_PATH", "/chat/completions").strip()
    openai_image_generations_path: str = os.getenv("OPENAI_IMAGE_GENERATIONS_PATH", "/images/generations").strip()
    url_blacklist: tuple[str, ...] = tuple(
        part.strip().lower()
        for part in os.getenv("URL_BLACKLIST", "localhost,127.0.0.1,0.0.0.0").split(",")
        if part.strip()
    )


def get_settings() -> Settings:
    return Settings()
