from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Any

from app.config import Settings


class QueryCacheService:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.query_cache_enabled
        self.ttl_seconds = max(1, settings.query_cache_ttl_seconds)
        self.max_entries = max(1, settings.query_cache_max_entries)
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def make_key(self, query: str, top_k: int, session_id: str | None = None) -> str:
        normalized = " ".join(query.lower().split())
        return f"{session_id or 'global'}:{top_k}:{normalized}"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        item = self._items.get(key)
        if not item:
            return None
        if time.time() - float(item["created_at"]) > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        item["hits"] += 1
        self._items.move_to_end(key)
        return copy.deepcopy(item["value"])

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._items[key] = {"created_at": time.time(), "hits": 0, "value": copy.deepcopy(value)}
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()
