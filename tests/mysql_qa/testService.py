from __future__ import annotations

from model_trian_classify.query_router import RouteDecision, RouteLabel
from mysql_qa.bm25_matcher import Bm25InnerProductMatcher
from mysql_qa.models import QuestionAnswer
from mysql_qa.service import MysqlQaService
from tests.mysql_qa.testMysqlClient import make_settings


class FakeRedisCache:
    def __init__(self, questions: list[QuestionAnswer]) -> None:
        self.questions = questions

    def get_questions(self) -> list[QuestionAnswer]:
        return self.questions


class FakeMysqlClient:
    def __init__(self, record: QuestionAnswer | None) -> None:
        self.record = record
        self.requested_id: str | None = None

    def get_answer(self, question_id: str) -> QuestionAnswer | None:
        self.requested_id = question_id
        return self.record


class FakeQueryRouter:
    def route(self, question: str) -> RouteDecision:
        return RouteDecision(
            label=RouteLabel.GENERAL_KNOWLEDGE,
            confidence=0.95,
            target_route="rag_qa",
            model_version="test",
        )


def testMatcherUsesSoftmaxNormalizedInnerProductScores() -> None:
    candidates = [
        QuestionAnswer(question_id="refund", question="如何申请退款"),
        QuestionAnswer(question_id="delivery", question="发货需要多久"),
    ]

    matches = Bm25InnerProductMatcher().match("退款怎么申请", candidates)

    assert matches[0].question.question_id == "refund"
    assert sum(match.confidence for match in matches) == 1.0


def testReturnsMysqlAnswerWhenConfidenceReachesThreshold() -> None:
    settings = make_settings().model_copy(update={"mysql_qa_threshold": 0.5})
    question = QuestionAnswer(question_id="refund", question="如何申请退款", answer="请在订单页申请退款。")
    mysql_client = FakeMysqlClient(question)
    service = MysqlQaService(settings, mysql_client, FakeRedisCache([question]))  # type: ignore[arg-type]

    result = service.answer("退款怎么申请")

    assert result.answer == "请在订单页申请退款。"
    assert result.route_to_rag_qa is False
    assert mysql_client.requested_id == "refund"


def testRoutesToRagQaWhenConfidenceIsInsufficient() -> None:
    settings = make_settings().model_copy(update={"mysql_qa_threshold": 0.99})
    candidates = [
        QuestionAnswer(question_id="refund", question="如何申请退款"),
        QuestionAnswer(question_id="delivery", question="发货需要多久"),
    ]
    service = MysqlQaService(settings, FakeMysqlClient(None), FakeRedisCache(candidates))  # type: ignore[arg-type]

    result = service.answer("退款怎么申请")

    assert result.answer is None
    assert result.route_to_rag_qa is True
    assert result.fallback_reason == "low_confidence"
    assert result.question_for_rag_qa == "退款怎么申请"


def testRoutesToRagQaWhenBm25HasNoOverlap() -> None:
    settings = make_settings().model_copy(update={"mysql_qa_threshold": 0.5})
    candidate = QuestionAnswer(question_id="refund", question="如何申请退款")
    service = MysqlQaService(settings, FakeMysqlClient(None), FakeRedisCache([candidate]))  # type: ignore[arg-type]

    result = service.answer("shipping status")

    assert result.route_to_rag_qa is True
    assert result.fallback_reason == "no_bm25_overlap"


def testUsesQueryRouterBeforeUnresolvedFaqIsSentToUnifiedRagQa() -> None:
    settings = make_settings().model_copy(update={"mysql_qa_threshold": 0.5})
    candidate = QuestionAnswer(question_id="refund", question="如何申请退款")
    service = MysqlQaService(
        settings,
        FakeMysqlClient(None),
        FakeRedisCache([candidate]),  # type: ignore[arg-type]
        query_router=FakeQueryRouter(),  # type: ignore[arg-type]
    )

    result = service.answer("shipping status")

    assert result.route_to_rag_qa is True
    assert result.question_for_rag_qa == "shipping status"
    assert result.router_label == "通用知识"
