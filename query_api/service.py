"""FAQ、分类、改写与 RAG 的统一问答编排。"""

from __future__ import annotations

from base.config import Settings
from base.logger import get_logger
from model_trian_classify.query_router import QueryRouter, RouteLabel
from mysql_qa.models import MysqlQaResult
from mysql_qa.service import MysqlQaService
from query_api.feature_flags import FeatureFlagService
from query_api.models import QueryAnswer, QueryCitation, WebCitation
from question_rewrite.models import RewriteResult
from question_rewrite.service import QuestionRewriteService
from rag_qa.service import RagQaService


class QueryAnswerService:
    """FAQ 命中即返回；未命中时按分类决定是否改写并统一进入 RAG。"""

    def __init__(
        self,
        app_settings: Settings,
        faq_service: MysqlQaService,
        query_router: QueryRouter,
        rewrite_service: QuestionRewriteService,
        rag_service: RagQaService,
        feature_flags: FeatureFlagService,
    ) -> None:
        self._settings = app_settings
        self._faq_service = faq_service
        self._query_router = query_router
        self._rewrite_service = rewrite_service
        self._rag_service = rag_service
        self._feature_flags = feature_flags
        self._logger = get_logger("query_api.service")

    def answer(self, question: str) -> QueryAnswer:
        """执行完整问答链路，并将不可恢复错误转换为客服电话。"""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        flags = self._feature_flags.current()
        faq_result = self._faq_service.answer(normalized_question) if flags.faq_enabled else MysqlQaResult(
            answer=None,
            confidence=0.0,
            matched_question=None,
            route_to_rag_qa=True,
            question_for_rag_qa=normalized_question,
        )
        if not flags.faq_enabled:
            self._logger.info("FAQ module disabled; routing directly past MySQL and Redis")
        if faq_result.answer is not None and not faq_result.route_to_rag_qa:
            self._logger.info("query answered by FAQ: confidence=%.4f", faq_result.confidence)
            return QueryAnswer(
                answer=faq_result.answer,
                source="faq",
                classification=None,
                citations=(),
                faq_confidence=faq_result.confidence,
            )

        classification: str | None = None
        classification_confidence: float | None = None
        classification_fallback: str | None = None
        if flags.classifier_enabled:
            try:
                decision = self._query_router.route(normalized_question)
                classification = decision.label.value
                classification_confidence = decision.confidence
            except Exception:
                self._logger.exception("Query classification failed; treating question as professional")
                classification = RouteLabel.PROFESSIONAL_CONSULTATION.value
                classification_confidence = 0.0
                classification_fallback = "classification_error"
        else:
            self._logger.info("Intent-classifier module disabled; routing directly to RAG")

        try:
            rewrite_result = (
                self._rewrite_service.rewrite(normalized_question, classification)
                if classification is not None
                else RewriteResult(normalized_question, (normalized_question,), (), 0, 0, 0)
            )
            rag_result = self._rag_service.answer(rewrite_result)
        except Exception:
            self._logger.exception("Query orchestration failed; returning customer service phone")
            return self._customer_service(faq_result, classification, classification_confidence)

        citations = tuple(
            QueryCitation(
                chunk_id=parent.chunk_id,
                source=parent.source,
                title_path=parent.title_path,
                text=parent.text,
            )
            for parent in rag_result.parents
        )
        web_citations = tuple(
            WebCitation(title=result.title, url=result.url, snippet=result.snippet)
            for result in rag_result.web_results
        )
        self._logger.info(
            "query answered by RAG: classification=%s citations=%s fallback=%s",
            classification,
            len(citations),
            rag_result.fallback_reason,
        )
        return QueryAnswer(
            answer=rag_result.answer,
            source="rag",
            classification=classification,
            citations=citations,
            faq_confidence=faq_result.confidence,
            classification_confidence=classification_confidence,
            fallback_reason=rag_result.fallback_reason or classification_fallback,
            web_citations=web_citations,
            web_search_used=bool(rag_result.web_results),
        )

    def _customer_service(
        self,
        faq_result: MysqlQaResult,
        classification: str | None,
        classification_confidence: float | None,
    ) -> QueryAnswer:
        return QueryAnswer(
            answer=f"客服电话：{self._settings.rag_customer_service_phone}",
            source="rag",
            classification=classification,
            citations=(),
            faq_confidence=faq_result.confidence,
            classification_confidence=classification_confidence,
            fallback_reason="query_orchestration_error",
        )
