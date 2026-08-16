"""混合检索、父块选择和基于原文的最终回答编排。"""

from __future__ import annotations

from base.config import Settings
from base.logger import get_logger
from question_rewrite.models import RewriteResult
from rag_qa.models import (
    AnswerModel,
    ParentChunk,
    ParentReranker,
    ParentRetriever,
    RagQaResult,
)


class RagQaService:
    """将改写结果转为带原文引用的答案或客服电话回退。"""

    def __init__(
        self,
        settings: Settings,
        retriever: ParentRetriever,
        reranker: ParentReranker,
        answer_model: AnswerModel,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._reranker = reranker
        self._answer_model = answer_model
        self._logger = get_logger("rag_qa.service")

    def answer(
        self,
        rewrite_result: RewriteResult,
    ) -> RagQaResult:
        """按父块数量执行电话回退、直接回答或 rerank 后回答。"""
        try:
            report = self._retriever.retrieve(rewrite_result.rag_queries)
            parents = report.parents
            if len(parents) >= self._settings.rag_final_parent_count:
                ranked_parents = self._reranker.rerank(rewrite_result.rag_queries, parents)
                parents = tuple(ranked_parents[: self._settings.rag_final_parent_count])
                self._logger.info("rag rerank selected parent_count=%s", len(parents))
            generation = self._answer_model.answer(rewrite_result.original_question, parents)
            if generation.text == "UNANSWERABLE":
                return self._customer_service("answer_not_grounded", parents)
            if not generation.text:
                return self._customer_service("empty_answer", parents)
            return RagQaResult(
                answer=generation.text,
                parents=parents,
                web_results=generation.web_results,
            )
        except Exception:
            self._logger.exception("rag_qa failed; returning customer service phone")
            return self._customer_service("rag_qa_error")

    def _customer_service(
        self, reason: str, parents: tuple[ParentChunk, ...] = ()
    ) -> RagQaResult:
        self._logger.info("rag_qa customer-service fallback: reason=%s", reason)
        return RagQaResult(
            answer=f"客服电话：{self._settings.rag_customer_service_phone}",
            parents=parents,
            fallback_reason=reason,
        )
