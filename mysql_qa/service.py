"""mysql_qa 命中决策及 RAG 路由边界。"""

from __future__ import annotations

from base.config import Settings
from base.logger import get_logger
from model_trian_classify.query_router import QueryRouter
from mysql_qa.bm25_matcher import Bm25InnerProductMatcher
from mysql_qa.models import MysqlQaResult
from mysql_qa.mysql_client import MysqlQaClient
from mysql_qa.redis_client import RedisQuestionCache


class MysqlQaService:
    """仅在置信度达到配置阈值时读取 MySQL 答案。"""

    def __init__(
        self,
        settings: Settings,
        mysql_client: MysqlQaClient,
        redis_cache: RedisQuestionCache,
        matcher: Bm25InnerProductMatcher | None = None,
        query_router: QueryRouter | None = None,
    ) -> None:
        self._settings = settings
        self._mysql_client = mysql_client
        self._redis_cache = redis_cache
        self._matcher = matcher or Bm25InnerProductMatcher()
        self._query_router = query_router
        self._logger = get_logger("mysql_qa.service")

    def answer(self, user_question: str) -> MysqlQaResult:
        """匹配 Redis 问题；低置信度或异常时路由到 rag_qa。"""
        if not user_question.strip():
            return self._route_to_rag_qa(user_question, "empty_question")
        try:
            candidates = self._redis_cache.get_questions()
            matches = self._matcher.match(user_question, candidates)
            if not matches:
                return self._route_to_rag_qa(user_question, "no_cached_question")
            best_match = matches[0]
            self._logger.info(
                "mysql_qa match: question_id=%s score=%.4f confidence=%.4f",
                best_match.question.question_id,
                best_match.score,
                best_match.confidence,
            )
            if best_match.score <= 0:
                return self._route_to_rag_qa(
                    user_question, "no_bm25_overlap", 0.0, best_match.question.question
                )
            if best_match.confidence <= self._settings.mysql_qa_threshold:
                return self._route_to_rag_qa(
                    user_question, "low_confidence", best_match.confidence, best_match.question.question
                )

            matched_question_id = best_match.question.question_id
            record = self._mysql_client.get_answer(matched_question_id)
            if record is None or record.answer is None:
                return self._route_to_rag_qa(
                    user_question,
                    "mysql_answer_not_found",
                    best_match.confidence,
                    best_match.question.question,
                )
            if record.question_id != matched_question_id:
                self._logger.error(
                    "mysql_qa answer ID mismatch: redis_question_id=%s mysql_question_id=%s",
                    matched_question_id,
                    record.question_id,
                )
                return self._route_to_rag_qa(
                    user_question,
                    "mysql_answer_id_mismatch",
                    best_match.confidence,
                    best_match.question.question,
                )
            return MysqlQaResult(
                answer=record.answer,
                confidence=best_match.confidence,
                matched_question=best_match.question.question,
                route_to_rag_qa=False,
            )
        except Exception:
            self._logger.exception("mysql_qa failed; routing question to rag_qa")
            return self._route_to_rag_qa(user_question, "mysql_qa_error")

    def _route_to_rag_qa(
        self,
        user_question: str,
        reason: str,
        confidence: float = 0.0,
        matched_question: str | None = None,
    ) -> MysqlQaResult:
        """先经查询分类器，再标记上层编排器要进入的节点。"""
        if self._query_router is None or not user_question.strip():
            self._logger.info("Routing question to rag_qa: reason=%s confidence=%.4f", reason, confidence)
            return MysqlQaResult(
                answer=None,
                confidence=confidence,
                matched_question=matched_question,
                route_to_rag_qa=True,
                question_for_rag_qa=user_question,
                fallback_reason=reason,
            )

        decision = self._query_router.route(user_question)
        route_to_rag_qa = decision.target_route in {"rag_qa", "web_rag"}
        self._logger.info(
            "FAQ unresolved route: reason=%s label=%s router_confidence=%.4f target=%s",
            reason,
            decision.label.value,
            decision.confidence,
            decision.target_route,
        )
        return MysqlQaResult(
            answer=None,
            confidence=confidence,
            matched_question=matched_question,
            route_to_rag_qa=route_to_rag_qa,
            question_for_rag_qa=user_question if route_to_rag_qa else None,
            question_for_general_qa=user_question if not route_to_rag_qa else None,
            fallback_reason=reason,
            router_label=decision.label.value,
            router_confidence=decision.confidence,
        )
