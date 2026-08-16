"""DuckDuckGo（DDGS）文本检索适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from base.config import Settings
from base.logger import get_logger
from rag_qa.models import WebSearchResult


class DuckDuckGoWebSearcher:
    """检索精简网页摘要；调用失败时由上层继续使用本地 RAG。"""

    def __init__(self, settings: Settings) -> None:
        self._max_results = settings.web_search_max_results
        self._region = settings.web_search_region
        self._timeout = settings.web_search_timeout_seconds
        self._logger = get_logger("web_search.duckduckgo")

    def search(self, question: str) -> tuple[WebSearchResult, ...]:
        from ddgs import DDGS

        try:
            raw_results: Sequence[dict[str, Any]] = DDGS(timeout=self._timeout).text(
                question,
                region=self._region,
                max_results=self._max_results,
            )
        except Exception:
            self._logger.exception("DuckDuckGo search failed")
            return ()

        results: list[WebSearchResult] = []
        for item in raw_results:
            title = str(item.get("title", "")).strip()
            url = str(item.get("href", "")).strip()
            snippet = str(item.get("body", "")).strip()
            if title and url and snippet:
                results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        self._logger.info("DuckDuckGo search completed: result_count=%s", len(results))
        return tuple(results)
