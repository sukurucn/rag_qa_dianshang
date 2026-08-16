from __future__ import annotations

from model_trian_classify.query_router import RouteDecision, RouteLabel
from mysql_qa.models import MysqlQaResult
from query_api.feature_flags import FeatureFlags
from query_api.service import QueryAnswerService
from question_rewrite.models import RewriteResult
from rag_qa.models import ParentChunk, RagQaResult
from tests.mysql_qa.testMysqlClient import make_settings


class FakeFaqService:
    def __init__(self, result: MysqlQaResult) -> None:
        self.result = result
        self.calls = 0

    def answer(self, question: str) -> MysqlQaResult:
        self.calls += 1
        return self.result


class FakeRouter:
    def __init__(self, label: RouteLabel) -> None:
        self.label = label
        self.calls = 0

    def route(self, question: str) -> RouteDecision:
        self.calls += 1
        return RouteDecision(self.label, 0.93, "rag_qa", "test")


class FakeRewriteService:
    def __init__(self) -> None:
        self.classifications: list[str] = []

    def rewrite(self, question: str, classification: str) -> RewriteResult:
        self.classifications.append(classification)
        queries = ("改写后的问题",) if classification == "专业咨询" else (question,)
        return RewriteResult(question, queries, (), 0, 0, 0)


class FakeRagService:
    def __init__(self) -> None:
        self.queries: tuple[str, ...] = ()

    def answer(self, rewrite_result: RewriteResult) -> RagQaResult:
        self.queries = rewrite_result.rag_queries
        parent = ParentChunk("parent-1", "course.md", "费用", "课程费用为 100 元")
        return RagQaResult("课程费用为 100 元。[来源：course.md / 费用]", (parent,))


class FakeFeatureFlags:
    def __init__(self, flags: FeatureFlags) -> None:
        self.flags = flags

    def current(self) -> FeatureFlags:
        return self.flags


def faq_hit() -> MysqlQaResult:
    return MysqlQaResult("FAQ 答案", 0.99, "标准问题", False)


def faq_miss() -> MysqlQaResult:
    return MysqlQaResult(None, 0.25, None, True, question_for_rag_qa="用户问题")


def make_service(
    faq_result: MysqlQaResult,
    label: RouteLabel,
    flags: FeatureFlags | None = None,
) -> tuple[QueryAnswerService, FakeFaqService, FakeRouter, FakeRewriteService, FakeRagService]:
    faq = FakeFaqService(faq_result)
    router = FakeRouter(label)
    rewriter = FakeRewriteService()
    rag = FakeRagService()
    service = QueryAnswerService(
        make_settings(),
        faq,  # type: ignore[arg-type]
        router,  # type: ignore[arg-type]
        rewriter,  # type: ignore[arg-type]
        rag,  # type: ignore[arg-type]
        FakeFeatureFlags(flags or FeatureFlags()),  # type: ignore[arg-type]
    )
    return service, faq, router, rewriter, rag


def testFaqHitReturnsImmediatelyWithoutClassificationOrRag() -> None:
    service, _, router, rewriter, rag = make_service(faq_hit(), RouteLabel.GENERAL_KNOWLEDGE)

    result = service.answer("退款怎么申请")

    assert result.answer == "FAQ 答案"
    assert result.source == "faq"
    assert router.calls == 0
    assert rewriter.classifications == []
    assert rag.queries == ()


def testGeneralKnowledgeUsesLocalRagBeforeAnswerAgentMayUseTools() -> None:
    service, _, _, rewriter, rag = make_service(faq_miss(), RouteLabel.GENERAL_KNOWLEDGE)

    result = service.answer("Python 是什么？")

    assert rewriter.classifications == ["通用知识"]
    assert rag.queries == ("Python 是什么？",)
    assert result.source == "rag"
    assert result.classification == "通用知识"
    assert result.web_search_used is False
    assert result.web_citations == ()
    assert result.citations[0].text == "课程费用为 100 元"


def testProfessionalConsultationUsesRewrittenQuestionForRag() -> None:
    service, _, _, rewriter, rag = make_service(
        faq_miss(), RouteLabel.PROFESSIONAL_CONSULTATION
    )

    result = service.answer("课程适合我吗？")

    assert rewriter.classifications == ["专业咨询"]
    assert rag.queries == ("改写后的问题",)
    assert result.classification == "专业咨询"


def testDisabledFaqSkipsFaqServiceAndUsesRag() -> None:
    service, faq, _, _, rag = make_service(
        faq_hit(),
        RouteLabel.PROFESSIONAL_CONSULTATION,
        FeatureFlags(faq_enabled=False, classifier_enabled=True),
    )

    result = service.answer("课程适合我吗？")

    assert faq.calls == 0
    assert result.source == "rag"
    assert rag.queries == ("改写后的问题",)


def testDisabledClassifierDirectlyUsesOriginalQuestionForMilvusRag() -> None:
    service, _, router, rewriter, rag = make_service(
        faq_miss(),
        RouteLabel.GENERAL_KNOWLEDGE,
        FeatureFlags(faq_enabled=True, classifier_enabled=False),
    )

    result = service.answer("Python 是什么？")

    assert router.calls == 0
    assert rewriter.classifications == []
    assert rag.queries == ("Python 是什么？",)
    assert result.classification is None
    assert result.web_search_used is False
