from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.request
from typing import Any

from app.config import Settings


class OpenAIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def enabled(self) -> bool:
        return bool(self.settings.openai_api_key and self.settings.openai_base_url)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        text = self.generate_text(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=True)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def generate_text(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str | None:
        if not self.enabled():
            return None
        payload: dict[str, Any] = {
            "model": self.settings.openai_text_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._post_json(
            self.settings.openai_chat_completions_path,
            payload,
            timeout=self.settings.openai_text_timeout_seconds,
        )
        if not response:
            return None
        choices = response.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            values = []
            for item in content:
                if isinstance(item, dict):
                    values.append(item.get("text", ""))
                else:
                    values.append(str(item))
            return "".join(values).strip()
        return str(content).strip()

    def generate_image(self, prompt: str, size: str, quality: str) -> dict[str, Any] | None:
        if not self.enabled():
            return None
        payload = {
            "model": self.settings.openai_image_model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
        }
        response = self._post_json(
            self.settings.openai_image_generations_path,
            payload,
            timeout=self.settings.openai_image_timeout_seconds,
        )
        if not response:
            return None
        data = response.get("data") or []
        if not data:
            return None
        first = data[0]
        return {
            "image_b64": first.get("b64_json"),
            "image_url": first.get("url"),
            "revised_prompt": first.get("revised_prompt", prompt) or prompt,
            "model": self.settings.openai_image_model,
        }

    def save_base64_image(self, image_b64: str, output_path: str) -> None:
        payload = base64.b64decode(image_b64)
        with open(output_path, "wb") as file:
            file.write(payload)

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any] | None:
        normalized_path = path if path.startswith("/") else f"/{path}"
        url = self.settings.openai_base_url.rstrip("/") + normalized_path
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"third-party api http {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"third-party api unreachable: {error}") from error
        except TimeoutError as error:
            raise RuntimeError(f"third-party api timeout after {timeout}s") from error
        except socket.timeout as error:
            raise RuntimeError(f"third-party api timeout after {timeout}s") from error
