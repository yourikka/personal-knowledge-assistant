from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter
from typing import Any

from app.config import Settings
from app.db import KnowledgeRepository
from app.services.text_utils import extract_keywords


class GraphExtractionService:
    ENTITY_PATTERNS = {
        "person": re.compile(r"(?:^|\n)(?:作者|提出者|创始人|负责人|研究者|人物)[:：][ \t]*([A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff ._-]{1,32})"),
        "organization": re.compile(r"(?:^|\n)(?:公司|组织|机构|团队|实验室)[:：][ \t]*([A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff ._-]{1,32})"),
        "technology": re.compile(r"(?:^|\n)(?:技术|框架|模型|工具|算法|协议)[:：][ \t]*([A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff ._+-]{1,32})"),
        "concept": re.compile(r"(?:^|\n)(?:概念|主题|关键词|方法)[:：][ \t]*([A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff ._-]{1,32})"),
    }
    ASCII_TECH_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-_./][A-Za-z0-9]+)*\b")
    ENTITY_STOPWORDS = {
        "agent",
        "api",
        "http",
        "json",
        "当前",
        "内容",
        "多个",
        "可以",
        "用于",
        "使用",
        "适合",
        "构建",
        "编排",
        "流程",
        "摘要",
        "问答",
        "模型",
        "测试",
        "文档",
        "知识库",
    }
    TYPE_PRIORITY = {
        "technology": 4,
        "organization": 3,
        "person": 2,
        "concept": 1,
    }

    def __init__(self, settings: Settings, repo: KnowledgeRepository) -> None:
        self.settings = settings
        self.repo = repo

    def enabled(self) -> bool:
        return self.settings.graph_enabled

    def build_for_document(
        self,
        document: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.enabled():
            return {"entities": [], "edges": []}

        text = "\n".join(
            [
                document.get("title", ""),
                document.get("summary", ""),
                " ".join(document.get("tags", [])),
                document.get("cleaned_text", "")[:3000],
            ]
        )
        chunk_lookup = self._first_chunk_by_entity(chunks)
        entities = self._extract_entities(text=text, chunk_lookup=chunk_lookup)
        edges = self._extract_edges(document_id=document["id"], entities=entities)
        self.repo.replace_document_graph(document_id=document["id"], entities=entities, edges=edges)
        return {"entities": entities, "edges": edges}

    def extract_query_entities(self, query: str) -> list[str]:
        entities = [match.group(0).strip() for match in self.ASCII_TECH_PATTERN.finditer(query)]
        entities.extend(extract_keywords(query, limit=6))
        return self._dedupe_names(entities)

    def related_documents(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        names = self.extract_query_entities(query)
        entities = self.repo.find_entities_by_names(names, limit=self.settings.graph_query_top_k)
        if not entities:
            return []
        entity_ids = [entity["id"] for entity in entities]
        documents = self.repo.graph_documents_for_entities(entity_ids, limit=limit or self.settings.graph_query_top_k)
        neighbors = self.repo.graph_neighbors(entity_ids, limit=self.settings.graph_query_top_k)
        neighbor_entity_ids = {
            edge["target_entity_id"] if edge["source_entity_id"] in entity_ids else edge["source_entity_id"]
            for edge in neighbors
        }
        if neighbor_entity_ids:
            documents.extend(
                self.repo.graph_documents_for_entities(
                    list(neighbor_entity_ids),
                    limit=max(1, (limit or self.settings.graph_query_top_k) // 2),
                )
            )
        return self._dedupe_documents(documents)

    def graph_view(self, document_id: str) -> dict[str, Any]:
        entities = self.repo.list_document_entities(document_id)
        edges = self.repo.list_document_graph_edges(document_id)
        return {
            "nodes": [
                {
                    "id": entity["id"],
                    "name": entity["name"],
                    "type": entity["entity_type"],
                    "mention_count": entity.get("mention_count"),
                }
                for entity in entities
            ],
            "edges": [
                {
                    "id": edge["id"],
                    "source": edge["source_entity_id"],
                    "target": edge["target_entity_id"],
                    "source_name": edge["source_name"],
                    "target_name": edge["target_name"],
                    "relation": edge["relation"],
                    "confidence": edge["confidence"],
                    "evidence_chunk_id": edge["evidence_chunk_id"],
                }
                for edge in edges
            ],
        }

    def _extract_entities(self, text: str, chunk_lookup: dict[str, str | None]) -> list[dict[str, Any]]:
        counts: Counter[tuple[str, str]] = Counter()
        explicit_labels: set[tuple[str, str]] = set()
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                name = self._clean_name(match.group(1))
                if self._valid_name(name):
                    counts[(name, entity_type)] += 3
                    explicit_labels.add((name.lower(), entity_type))

        for tag in extract_keywords(text, limit=12):
            name = self._clean_name(tag)
            if self._valid_name(name):
                counts[(name, "concept")] += 1

        for match in self.ASCII_TECH_PATTERN.finditer(text):
            name = self._clean_name(match.group(0))
            if self._valid_name(name):
                counts[(name, "technology")] += 1

        best_by_name: dict[str, tuple[str, str, int]] = {}
        for (name, entity_type), count in counts.items():
            key = name.lower()
            current = best_by_name.get(key)
            if current is None:
                best_by_name[key] = (name, entity_type, count)
                continue
            _, current_type, current_count = current
            current_rank = (
                int((key, current_type) in explicit_labels),
                current_count,
                self.TYPE_PRIORITY.get(current_type, 0),
            )
            candidate_rank = (
                int((key, entity_type) in explicit_labels),
                count,
                self.TYPE_PRIORITY.get(entity_type, 0),
            )
            if candidate_rank > current_rank:
                best_by_name[key] = (name, entity_type, count)

        ranked = sorted(
            best_by_name.values(),
            key=lambda item: (-item[2], -self.TYPE_PRIORITY.get(item[1], 0), item[0]),
        )
        entities = []
        for name, entity_type, count in ranked[:16]:
            entity_id = self._entity_id(name=name, entity_type=entity_type)
            entities.append(
                {
                    "id": entity_id,
                    "name": name,
                    "entity_type": entity_type,
                    "aliases": [name.lower()] if name.lower() != name else [],
                    "mention_count": count,
                    "first_chunk_id": chunk_lookup.get(name.lower()),
                    "metadata": {"extractor": "heuristic"},
                }
            )
        return entities

    def _extract_edges(self, document_id: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        edges = []
        for left, right in itertools.combinations(entities[:8], 2):
            relation = self._relation(left["entity_type"], right["entity_type"])
            confidence = min(0.95, 0.42 + (left["mention_count"] + right["mention_count"]) * 0.04)
            edge_id = self._edge_id(document_id, left["id"], right["id"], relation)
            edges.append(
                {
                    "id": edge_id,
                    "source_entity_id": left["id"],
                    "target_entity_id": right["id"],
                    "relation": relation,
                    "confidence": round(confidence, 2),
                    "evidence_chunk_id": left.get("first_chunk_id") or right.get("first_chunk_id"),
                    "metadata": {"extractor": "heuristic"},
                }
            )
        return edges[:24]

    def _first_chunk_by_entity(self, chunks: list[dict[str, Any]]) -> dict[str, str | None]:
        lookup: dict[str, str | None] = {}
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            for keyword in extract_keywords(chunk.get("text", ""), limit=8):
                key = keyword.strip().lower()
                if key and key in text and key not in lookup:
                    lookup[key] = chunk["id"]
            for match in self.ASCII_TECH_PATTERN.finditer(chunk.get("text", "")):
                key = match.group(0).strip().lower()
                if key and key not in lookup:
                    lookup[key] = chunk["id"]
        return lookup

    def _relation(self, left_type: str, right_type: str) -> str:
        pair = {left_type, right_type}
        if "person" in pair and "technology" in pair:
            return "proposes_or_uses"
        if "organization" in pair and "technology" in pair:
            return "develops_or_uses"
        if "concept" in pair and "technology" in pair:
            return "explains"
        return "related_to"

    def _entity_id(self, name: str, entity_type: str) -> str:
        digest = hashlib.sha256(f"{entity_type}:{name.lower()}".encode("utf-8")).hexdigest()[:20]
        return f"ent-{digest}"

    def _edge_id(self, document_id: str, source_id: str, target_id: str, relation: str) -> str:
        ordered = "|".join(sorted([source_id, target_id]))
        digest = hashlib.sha256(f"{document_id}:{ordered}:{relation}".encode("utf-8")).hexdigest()[:24]
        return f"edge-{digest}"

    def _clean_name(self, value: str) -> str:
        candidate = re.split(r"[，。；;、,.!?！？\n\r\t]", value.strip(), maxsplit=1)[0]
        candidate = re.sub(r"\s+", " ", candidate.strip(" \t\r\n，。；;：:（）()[]【】"))
        return candidate

    def _valid_name(self, value: str) -> bool:
        if len(value) < self.settings.graph_min_entity_length or len(value) > 40:
            return False
        lower = value.lower()
        if lower in self.ENTITY_STOPWORDS:
            return False
        if re.fullmatch(r"[a-z0-9_-]+", lower) and len(lower) < 4:
            return False
        if any(word in lower for word in ("可以", "用于", "适合", "不会", "当前")):
            return False
        if re.fullmatch(r"\d+", value):
            return False
        if len(re.findall(r"[\u4e00-\u9fff]", value)) > 8:
            return False
        if value.count(" ") > 2:
            return False
        if re.fullmatch(r"[\u4e00-\u9fff]+", value) and len(value) > 6:
            return False
        return any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in value)

    def _dedupe_names(self, names: list[str]) -> list[str]:
        seen = set()
        deduped = []
        for name in names:
            cleaned = self._clean_name(name)
            key = cleaned.lower()
            if self._valid_name(cleaned) and key not in seen:
                seen.add(key)
                deduped.append(cleaned)
        return deduped

    def _dedupe_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        deduped = []
        for document in documents:
            if document["id"] in seen:
                continue
            seen.add(document["id"])
            deduped.append(document)
        return deduped
