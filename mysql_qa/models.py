"""mysql_qa 模块的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionAnswer:
    """MySQL 中持久化的一条标准问题和答案。"""

    question_id: str
    question: str
    answer: str | None = None


@dataclass(frozen=True)
class QaWriteResult:
    """一批 QA 写操作的审计结果。"""

    question_ids: tuple[str, ...]
    created_count: int
    updated_count: int


@dataclass(frozen=True)
class MysqlQaResult:
    """一次 MySQL QA 决策的结果，供上层编排器决定下一节点。"""

    answer: str | None
    confidence: float
    matched_question: str | None
    route_to_rag_qa: bool
    question_for_rag_qa: str | None = None
    question_for_general_qa: str | None = None
    fallback_reason: str | None = None
    router_label: str | None = None
    router_confidence: float | None = None
