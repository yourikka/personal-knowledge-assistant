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
