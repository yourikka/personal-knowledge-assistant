from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from app.config import Settings
from app.services.text_utils import make_hash_embedding


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def enabled(self) -> bool:
        return (
            self.settings.embedding_provider == "zhipu"
            and bool(self.settings.embedding_api_key)
            and bool(self.settings.embedding_base_url)
        )

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            dims = self.settings.embedding_dimensions if self.enabled() else 128
            return [0.0] * dims

        if not self.enabled():
            return make_hash_embedding(text)

        payload = {
            "model": self.settings.embedding_model,
            "input": text,
            "dimensions": self.settings.embedding_dimensions,
        }
        try:
            response = self._post_json(
                self.settings.embedding_path,
                payload,
                timeout=self.settings.embedding_timeout_seconds,
            )
        except Exception:
            return make_hash_embedding(text, dims=self.settings.embedding_dimensions)

        data = response.get("data") or []
        if not data:
            return make_hash_embedding(text, dims=self.settings.embedding_dimensions)
        embedding = data[0].get("embedding") or []
        if not embedding:
            return make_hash_embedding(text, dims=self.settings.embedding_dimensions)
        return [float(value) for value in embedding]

    def _post_json(self, path: str, payload: dict, timeout: int) -> dict:
        normalized_path = path if path.startswith("/") else f"/{path}"
        url = self.settings.embedding_base_url.rstrip("/") + normalized_path
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.embedding_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"embedding api http {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"embedding api unreachable: {error}") from error
        except TimeoutError as error:
            raise RuntimeError(f"embedding api timeout after {timeout}s") from error
        except socket.timeout as error:
            raise RuntimeError(f"embedding api timeout after {timeout}s") from error
