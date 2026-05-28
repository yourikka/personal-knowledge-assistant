from __future__ import annotations

import io
import mimetypes
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from .text_utils import normalize_document_text

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
    parser_metadata: dict[str, Any] = {}
    if source_type == "pdf":
        text, parser_metadata = parse_pdf_document(raw_bytes)
    elif source_type == "image":
        text, parser_metadata = parse_image_document(raw_bytes)
    else:
        parsers = {
            "html": parse_html,
            "url": parse_html,
            "markdown": parse_markdown,
            "text": parse_plain_text,
        }
        parser = parsers.get(source_type, parse_plain_text)
        text = parser(raw_bytes)

    extra = extract_metadata_from_text(text)
    extra.update(parser_metadata)
    extra["parsed_chars"] = len(text)
    extra["parser"] = source_type
    merged_metadata = {**metadata, **extra}
    return normalize_document_text(text), merged_metadata


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
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            lines.append(line.strip())
            continue
        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = re.sub(r"^\s{0,3}[-*+]\s+", "", line)
        line = re.sub(r"^\s{0,3}\d+[.)]\s+", "", line)
        lines.append(line)
    return "\n".join(lines)


def parse_plain_text(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8", errors="ignore")


def parse_pdf(raw_bytes: bytes) -> str:
    text, _ = parse_pdf_document(raw_bytes)
    return text


def parse_pdf_document(raw_bytes: bytes) -> tuple[str, dict[str, Any]]:
    pages: list[str]
    if PdfReader is None:
        pages = [raw_bytes.decode("utf-8", errors="ignore")]
    else:
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception:
            pages = [raw_bytes.decode("utf-8", errors="ignore")]

    structured_pages = []
    headings: list[str] = []
    table_count = 0
    for index, page_text in enumerate(pages, start=1):
        structured, page_headings, page_tables = restore_document_structure(page_text)
        headings.extend(page_headings)
        table_count += page_tables
        structured_pages.append(f"## Page {index}\n{structured}".strip())

    text = "\n\n".join(page for page in structured_pages if page.strip())
    return text, {
        "page_count": len(pages),
        "structure": {
            "headings": headings[:20],
            "table_count": table_count,
            "layout": "page_heading_table_heuristic",
        },
    }


def parse_image_ocr(raw_bytes: bytes) -> str:
    text, _ = parse_image_document(raw_bytes)
    return text


def parse_image_document(raw_bytes: bytes) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "ocr_enabled": bool(Image is not None and pytesseract is not None),
        "ocr_languages": "chi_sim+eng",
    }
    if Image is None or pytesseract is None:
        return "OCR 未启用：缺少 Pillow 或 pytesseract。", metadata
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        metadata.update(
            {
                "image_width": image.width,
                "image_height": image.height,
                "image_mode": image.mode,
            }
        )
        text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        metadata["image_summary"] = f"图片尺寸 {image.width}x{image.height}，OCR 提取 {len(text.strip())} 字符。"
        return f"[Image OCR]\n{text.strip()}".strip(), metadata
    except Exception:
        metadata["ocr_error"] = "OCR 识别失败。"
        return "OCR 识别失败。", metadata


def restore_document_structure(text: str) -> tuple[str, list[str], int]:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()]
    restored = []
    headings = []
    table_count = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            restored.append("")
            continue
        if is_table_like_line(stripped):
            columns = [part.strip() for part in re.split(r"\s{2,}|\t+", stripped) if part.strip()]
            if len(columns) >= 2:
                restored.append("| " + " | ".join(columns) + " |")
                table_count += 1
                continue
        if is_heading_like_line(stripped, next_line=lines[index + 1].strip() if index + 1 < len(lines) else ""):
            heading = stripped.lstrip("# ").strip()
            headings.append(heading)
            restored.append(f"# {heading}")
            continue
        restored.append(stripped)
    return "\n".join(restored), headings, table_count


def is_table_like_line(line: str) -> bool:
    return bool(re.search(r"\S\s{2,}\S", line) or "\t" in line)


def is_heading_like_line(line: str, next_line: str = "") -> bool:
    if line.startswith("#"):
        return True
    if len(line) > 80 or line.endswith(("。", ".", "！", "？", ";", "；", "：", ":")):
        return False
    if re.match(r"^\d+(\.\d+)*\s+\S+", line):
        return True
    return bool(next_line and len(next_line) > len(line) * 1.8)


def extract_metadata_from_text(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = lines[0][:120] if lines else "未命名文档"
    return {"detected_title": title}
