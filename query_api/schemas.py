"""用户问答 HTTP 接口 schema。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class WebCitationResponse(BaseModel):
    """网络检索中使用的一条网页摘要。"""

    title: str
    url: str
    snippet: str


class QueryResponse(BaseModel):
    """FAQ 与 RAG 共用的 HTTP 响应。"""

    answer: str
    source: Literal["faq", "rag"]
    classification: str | None
    citations: list[QueryCitationResponse]
    faq_confidence: float
    classification_confidence: float | None = None
    fallback_reason: str | None = None
    web_citations: list[WebCitationResponse] = Field(default_factory=list)
    web_search_used: bool = False


class FeatureFlagsResponse(BaseModel):
    """前端可控制的问答模块开关。"""

    faq_enabled: bool
    classifier_enabled: bool


class FeatureFlagsUpdateRequest(BaseModel):
    """局部修改至少一个模块开关。"""

    faq_enabled: bool | None = None
    classifier_enabled: bool | None = None

    @model_validator(mode="after")
    def requires_a_change(self) -> FeatureFlagsUpdateRequest:
        if self.faq_enabled is None and self.classifier_enabled is None:
            raise ValueError("at least one feature flag must be supplied")
        return self
