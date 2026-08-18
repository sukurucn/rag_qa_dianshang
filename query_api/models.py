"""统一用户问答入口的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QueryCitation:
    """最终回答采用的一段可追溯父块原文。"""

    chunk_id: str
    source: str
    title_path: str
    text: str


@dataclass(frozen=True)
class WebCitation:
    """网络检索使用的一条网页摘要。"""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class QueryAnswer:
    """FAQ 或 RAG 统一返回的问答结果。"""

    answer: str
    source: Literal["faq", "rag"]
    classification: str | None
    citations: tuple[QueryCitation, ...]
    faq_confidence: float
    classification_confidence: float | None = None
    fallback_reason: str | None = None
    web_citations: tuple[WebCitation, ...] = ()
    web_search_used: bool = False
