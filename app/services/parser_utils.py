from __future__ import annotations

import io
import mimetypes
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from .text_utils import normalize_whitespace

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


def validate_url(url: str, blacklist: tuple[str, ...]) -> None:
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL 仅支持 http/https。")
    if hostname in blacklist:
        raise ValueError(f"URL 域名 `{hostname}` 在黑名单中。")


def fetch_url_content(url: str, enable_playwright: bool) -> tuple[bytes, dict[str, Any]]:
    started_at = time.time()
    if enable_playwright:
        rendered = fetch_url_with_playwright(url)
        if rendered is not None:
            payload = rendered.encode("utf-8")
            return payload, {
                "content_type": "text/html; charset=utf-8",
                "fetched_by": "playwright",
                "bytes": len(payload),
                "fetch_ms": int((time.time() - started_at) * 1000),
            }

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "personal-knowledge-assistant/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = response.read()
        content_type = response.headers.get_content_type()
    return payload, {
        "content_type": content_type,
        "fetched_by": "urllib",
        "bytes": len(payload),
        "fetch_ms": int((time.time() - started_at) * 1000),
    }


def fetch_url_with_playwright(url: str) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None


def read_source_payload(source_type: str, source: str, enable_playwright: bool, blacklist: tuple[str, ...]) -> tuple[bytes, str, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if source_type == "url":
        validate_url(source, blacklist)
        raw_bytes, metadata = fetch_url_content(source, enable_playwright)
        return raw_bytes, source, metadata

    if source_type in {"pdf", "image"}:
        with open(source, "rb") as file:
            raw_bytes = file.read()
        metadata["content_type"] = mimetypes.guess_type(source)[0] or "application/octet-stream"
        metadata["bytes"] = len(raw_bytes)
        metadata["filename"] = os.path.basename(source)
        return raw_bytes, os.path.abspath(source), metadata

    if os.path.exists(source):
        with open(source, "rb") as file:
            raw_bytes = file.read()
        metadata["content_type"] = mimetypes.guess_type(source)[0] or "text/plain"
        metadata["bytes"] = len(raw_bytes)
        metadata["filename"] = os.path.basename(source)
        return raw_bytes, os.path.abspath(source), metadata

    raw_bytes = source.encode("utf-8")
    metadata["bytes"] = len(raw_bytes)
    return raw_bytes, "inline://content", metadata


def parse_content(source_type: str, raw_bytes: bytes, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    parsers = {
        "html": parse_html,
        "url": parse_html,
        "markdown": parse_markdown,
        "text": parse_plain_text,
        "pdf": parse_pdf,
        "image": parse_image_ocr,
    }
    parser = parsers.get(source_type, parse_plain_text)
    text = parser(raw_bytes)
    extra = extract_metadata_from_text(text)
    extra["parsed_chars"] = len(text)
    extra["parser"] = source_type
    merged_metadata = {**metadata, **extra}
    return normalize_whitespace(text), merged_metadata


def parse_html(raw_bytes: bytes) -> str:
    html = raw_bytes.decode("utf-8", errors="ignore")
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        body = soup.get_text("\n", strip=True)
        return "\n".join(filter(None, [title, body]))
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    return re.sub(r"<[^>]+>", " ", html)


def parse_markdown(raw_bytes: bytes) -> str:
    text = raw_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r"`{1,3}.*?`{1,3}", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^[#>\-\*\d\.\s]+", "", text, flags=re.MULTILINE)
    return text


def parse_plain_text(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8", errors="ignore")


def parse_pdf(raw_bytes: bytes) -> str:
    if PdfReader is None:
        return raw_bytes.decode("utf-8", errors="ignore")
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return raw_bytes.decode("utf-8", errors="ignore")


def parse_image_ocr(raw_bytes: bytes) -> str:
    if Image is None or pytesseract is None:
        return "OCR 未启用：缺少 Pillow 或 pytesseract。"
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        return pytesseract.image_to_string(image, lang="chi_sim+eng")
    except Exception:
        return "OCR 识别失败。"


def extract_metadata_from_text(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0][:120] if lines else "未命名文档"
    return {"detected_title": title}
