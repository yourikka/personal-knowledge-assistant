from __future__ import annotations

import re
from typing import Any

from app.services.text_utils import normalize_whitespace, tokenize


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
SENTENCE_RE = re.compile(r"[^。！？!?；;.\n]+[。！？!?；;.]?|[^\n]+")


class DocumentChunker:
    def __init__(
        self,
        target_chars: int = 900,
        overlap_chars: int = 160,
        min_chars: int = 180,
        max_chars: int = 1400,
    ) -> None:
        self.target_chars = max(300, target_chars)
        self.overlap_chars = max(0, min(overlap_chars, self.target_chars // 2))
        self.min_chars = max(60, min(min_chars, self.target_chars))
        self.max_chars = max(self.target_chars, max_chars)

    def chunk(self, document_id: str, text: str) -> list[dict[str, Any]]:
        source = text.strip()
        if not source:
            return []

        units = self._semantic_units(source)
        ranges = self._merge_units(units)
        chunks = []
        for index, (start, end, heading_path) in enumerate(ranges):
            chunk_text = normalize_whitespace(source[start:end])
            if not chunk_text:
                continue
            chunks.append(
                {
                    "id": f"{document_id}:chunk:{len(chunks):04d}",
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "text": chunk_text,
                    "char_start": start,
                    "char_end": end,
                    "metadata": {
                        "heading_path": heading_path,
                        "heading": heading_path[-1] if heading_path else "全文",
                        "char_count": len(chunk_text),
                        "token_count": len(tokenize(chunk_text)),
                    },
                }
            )
        return chunks

    def sections_from_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not chunks:
            return []

        sections: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        current_heading: tuple[str, ...] | None = None

        for chunk in sorted(chunks, key=lambda item: item["chunk_index"]):
            heading_path = tuple(chunk.get("metadata", {}).get("heading_path") or [])
            if current and heading_path != current_heading:
                sections.append(self._section_from_chunks(document_id, len(sections), current, current_heading or ()))
                current = []
            current.append(chunk)
            current_heading = heading_path

        if current:
            sections.append(self._section_from_chunks(document_id, len(sections), current, current_heading or ()))
        return sections

    def sections(self, document_id: str, text: str) -> list[dict[str, Any]]:
        source = text.strip()
        if not source:
            return []

        units = self._semantic_units(source)
        if not units:
            return []

        sections: list[dict[str, Any]] = []
        current: list[dict[str, Any]] = []
        current_heading: tuple[str, ...] | None = None
        for unit in units:
            heading_path = tuple(unit.get("heading_path") or [])
            if current and heading_path != current_heading:
                sections.append(self._section_from_units(document_id, len(sections), source, current, current_heading or ()))
                current = []
            current.append(unit)
            current_heading = heading_path

        if current:
            sections.append(self._section_from_units(document_id, len(sections), source, current, current_heading or ()))
        return sections

    def _section_from_units(
        self,
        document_id: str,
        section_index: int,
        source: str,
        units: list[dict[str, Any]],
        heading_path: tuple[str, ...],
    ) -> dict[str, Any]:
        heading = heading_path[-1] if heading_path else "全文"
        start = units[0]["start"]
        end = units[-1]["end"]
        text = normalize_whitespace(source[start:end])
        return {
            "id": f"{document_id}:section:{section_index:04d}",
            "document_id": document_id,
            "section_index": section_index,
            "heading": heading,
            "heading_path": list(heading_path),
            "text": text,
            "char_start": start,
            "char_end": end,
            "metadata": {
                "unit_count": len(units),
                "char_count": len(text),
            },
        }

    def _section_from_chunks(
        self,
        document_id: str,
        section_index: int,
        chunks: list[dict[str, Any]],
        heading_path: tuple[str, ...],
    ) -> dict[str, Any]:
        heading = heading_path[-1] if heading_path else "全文"
        text = normalize_whitespace("\n\n".join(chunk["text"] for chunk in chunks))
        return {
            "id": f"{document_id}:section:{section_index:04d}",
            "document_id": document_id,
            "section_index": section_index,
            "heading": heading,
            "heading_path": list(heading_path),
            "text": text,
            "char_start": min(chunk["char_start"] for chunk in chunks),
            "char_end": max(chunk["char_end"] for chunk in chunks),
            "metadata": {
                "chunk_ids": [chunk["id"] for chunk in chunks],
                "chunk_count": len(chunks),
                "char_count": len(text),
            },
        }

    def _semantic_units(self, text: str) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        heading_stack: list[tuple[int, str]] = []
        position = 0

        for paragraph_match in re.finditer(r"\S[\s\S]*?(?=\n{2,}|\Z)", text):
            paragraph = paragraph_match.group(0).strip()
            start = paragraph_match.start()
            if not paragraph:
                continue

            heading = HEADING_RE.match(paragraph)
            if heading:
                level = len(heading.group(1))
                title = normalize_whitespace(heading.group(2))
                heading_stack = [(old_level, old_title) for old_level, old_title in heading_stack if old_level < level]
                heading_stack.append((level, title))
                units.append(
                    {
                        "start": start,
                        "end": paragraph_match.end(),
                        "text": title,
                        "heading_path": [item[1] for item in heading_stack],
                    }
                )
                position = paragraph_match.end()
                continue

            heading_path = [item[1] for item in heading_stack]
            if len(paragraph) <= self.max_chars:
                units.append(
                    {
                        "start": start,
                        "end": paragraph_match.end(),
                        "text": paragraph,
                        "heading_path": heading_path,
                    }
                )
            else:
                units.extend(self._split_long_paragraph(text, start, paragraph, heading_path))
            position = paragraph_match.end()

        if not units and position < len(text):
            units.extend(self._split_long_paragraph(text, 0, text, []))
        return units

    def _split_long_paragraph(
        self,
        source: str,
        base_start: int,
        paragraph: str,
        heading_path: list[str],
    ) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for match in SENTENCE_RE.finditer(paragraph):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            start = base_start + match.start()
            end = base_start + match.end()
            if len(sentence) <= self.max_chars:
                units.append({"start": start, "end": end, "text": sentence, "heading_path": heading_path})
                continue
            units.extend(self._hard_split(source, start, end, heading_path))
        return units

    def _hard_split(
        self,
        source: str,
        start: int,
        end: int,
        heading_path: list[str],
    ) -> list[dict[str, Any]]:
        units = []
        cursor = start
        step = max(1, self.max_chars - self.overlap_chars)
        while cursor < end:
            part_end = min(end, cursor + self.max_chars)
            units.append(
                {
                    "start": cursor,
                    "end": part_end,
                    "text": source[cursor:part_end],
                    "heading_path": heading_path,
                }
            )
            if part_end >= end:
                break
            cursor += step
        return units

    def _merge_units(self, units: list[dict[str, Any]]) -> list[tuple[int, int, list[str]]]:
        ranges: list[tuple[int, int, list[str]]] = []
        current: list[dict[str, Any]] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit["text"])
            heading_changed = current and unit.get("heading_path") != current[-1].get("heading_path")
            should_flush = current and current_len >= self.min_chars and current_len + unit_len > self.target_chars
            should_flush = should_flush or (current and current_len + unit_len > self.max_chars)
            should_flush = should_flush or bool(heading_changed)
            if should_flush:
                ranges.append(self._range_from_units(current))
                current = self._overlap_tail(current)
                current_len = sum(len(item["text"]) for item in current)
                if heading_changed:
                    current = []
                    current_len = 0
                if current and current_len + unit_len > self.max_chars:
                    current = []
                    current_len = 0

            current.append(unit)
            current_len += unit_len

        if current:
            if ranges and current_len < self.min_chars:
                previous_start, _, previous_heading = ranges[-1]
                current_heading = current[-1].get("heading_path") or previous_heading
                if current_heading == previous_heading:
                    ranges.pop()
                    ranges.append((previous_start, current[-1]["end"], current_heading))
                else:
                    ranges.append(self._range_from_units(current))
            else:
                ranges.append(self._range_from_units(current))

        return ranges

    def _range_from_units(self, units: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
        heading_path: list[str] = []
        for unit in units:
            if unit.get("heading_path"):
                heading_path = unit["heading_path"]
        return units[0]["start"], units[-1]["end"], heading_path

    def _overlap_tail(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.overlap_chars <= 0:
            return []
        tail: list[dict[str, Any]] = []
        total = 0
        for unit in reversed(units):
            if total >= self.overlap_chars:
                break
            tail.insert(0, unit)
            total += len(unit["text"])
        return tail
