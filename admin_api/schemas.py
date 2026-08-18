"""管理 API 的请求与响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class QaPayload(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    answer: str = Field(min_length=1, max_length=50_000)


class QaDeleteRequest(BaseModel):
    question_ids: list[str] = Field(min_length=1, max_length=1_000)


class QaItem(BaseModel):
    question_id: str
    question: str
    answer: str


class QaWriteResponse(BaseModel):
    question_ids: list[str]
    created_count: int
    updated_count: int
    deleted_count: int = 0
    redis_warmed_count: int


class QaListResponse(BaseModel):
    items: list[QaItem]
    limit: int
    offset: int


class IngestionJobResponse(BaseModel):
    job_id: str
    original_filename: str
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "DELETED"]
    document_id: str | None
    stored_chunks: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
