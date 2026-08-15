"""用户问答 HTTP 接口 schema。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """一次用户提问。"""

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        """拒绝仅包含空白字符的问题。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class QueryCitationResponse(BaseModel):
    """回答引用的父块原文。"""

    chunk_id: str
    source: str
    title_path: str
    text: str


class QueryResponse(BaseModel):
    """FAQ 与 RAG 共用的 HTTP 响应。"""

    answer: str
    source: Literal["faq", "rag"]
    classification: str | None
    citations: list[QueryCitationResponse]
    faq_confidence: float
    classification_confidence: float | None = None
    fallback_reason: str | None = None
