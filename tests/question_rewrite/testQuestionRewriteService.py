from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from question_rewrite.langchain_client import LangChainRewriteModel
from question_rewrite.models import RewriteModelResponse, RewriteStrategy
from question_rewrite.service import PROFESSIONAL_CONSULTATION, QuestionRewriteService
from tests.mysql_qa.testMysqlClient import make_settings


class FakeRewriteModel:
    def __init__(self, responses: list[RewriteModelResponse | Exception]) -> None:
        self._responses = responses
        self.requests: list[tuple[BaseMessage, ...]] = []

    def invoke(self, messages: Sequence[BaseMessage]) -> RewriteModelResponse:
        self.requests.append(tuple(messages))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    def __init__(self, response: AIMessage) -> None:
        self._response = response
        self.request: list[BaseMessage] = []

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.request = messages
        return self._response

    def get_num_tokens_from_messages(self, messages: list[BaseMessage]) -> int:
        return 42


def response(
    strategy: RewriteStrategy,
    *,
    rewritten_question: str | None = None,
    hypothetical_answer: str | None = None,
    subquestions: tuple[str, ...] = (),
) -> RewriteModelResponse:
    return RewriteModelResponse(
        strategy=strategy,
        reason="用于测试的简短依据",
        rewritten_question=rewritten_question,
        hypothetical_answer=hypothetical_answer,
        subquestions=subquestions,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
    )


def make_service(model: FakeRewriteModel, max_rounds: int = 3) -> QuestionRewriteService:
    settings = make_settings().model_copy(update={"rewrite_max_rounds": max_rounds})
    return QuestionRewriteService(settings, model)


def testBypassesLlmForNonProfessionalQuestion() -> None:
    model = FakeRewriteModel([])

    result = make_service(model).rewrite("什么是 Python？", "通用知识")

    assert result.rag_queries == ("什么是 Python？",)
    assert result.strategy_traces == ()
    assert model.requests == []


def testDenoiseThenHydeKeepsHistoryReadOnly() -> None:
    model = FakeRewriteModel(
        [
            response(RewriteStrategy.DENOISE, rewritten_question="AI课程收费是多少？"),
            response(RewriteStrategy.HYDE, hypothetical_answer="AI课程费用和付款方式说明。"),
        ]
    )
    history = [HumanMessage(content="我正在了解课程")]

    result = make_service(model).rewrite("那个 AI 课啊，收费大概多少呀呀？", PROFESSIONAL_CONSULTATION, history)

    assert result.rag_queries == ("AI课程收费是多少？", "AI课程费用和付款方式说明。")
    assert [trace.strategy for trace in result.strategy_traces] == [
        RewriteStrategy.DENOISE,
        RewriteStrategy.HYDE,
    ]
    assert result.max_context_tokens == 100
    assert result.total_input_tokens == 200
    assert len(history) == 1
    assert model.requests[0][1] is history[0]
    assert model.requests[1][1] is history[0]


def testDirectStrategyUsesOriginalQuestionForRag() -> None:
    model = FakeRewriteModel([response(RewriteStrategy.DIRECT)])

    result = make_service(model).rewrite("AI课程学费是多少？", PROFESSIONAL_CONSULTATION)

    assert result.rag_queries == ("AI课程学费是多少？",)
    assert result.strategy_traces[0].strategy is RewriteStrategy.DIRECT


def testTruncatesSplitQuestionsToConfiguredLimit() -> None:
    model = FakeRewriteModel(
        [
            response(
                RewriteStrategy.SPLIT,
                subquestions=("问题1", "问题2", "问题3", "问题4", "问题5", "问题6", "问题7"),
            )
        ]
    )

    result = make_service(model).rewrite("课程费用、地点和时间分别是什么？", PROFESSIONAL_CONSULTATION)

    assert result.rag_queries == ("问题1", "问题2", "问题3", "问题4", "问题5", "问题6")
    assert result.fallback_reason == "subquestions_truncated"


def testUsesThirdRoundRewriteWhenMaxRoundsReached() -> None:
    model = FakeRewriteModel(
        [
            response(RewriteStrategy.DENOISE, rewritten_question="第一次改写"),
            response(RewriteStrategy.DENOISE, rewritten_question="第二次改写"),
            response(RewriteStrategy.DENOISE, rewritten_question="第三次改写"),
        ]
    )

    result = make_service(model).rewrite("原始问题", PROFESSIONAL_CONSULTATION)

    assert result.rag_queries == ("第三次改写",)
    assert result.fallback_reason == "max_rounds"
    assert len(model.requests) == 3


def testUsesLatestRewriteWhenLaterModelCallFails() -> None:
    model = FakeRewriteModel(
        [
            response(RewriteStrategy.DENOISE, rewritten_question="最近成功的改写"),
            RuntimeError("LLM unavailable"),
        ]
    )

    result = make_service(model).rewrite("原始问题", PROFESSIONAL_CONSULTATION)

    assert result.rag_queries == ("最近成功的改写",)
    assert result.fallback_reason == "rewrite_error"
    assert len(result.strategy_traces) == 1


def testLangChainAdapterParsesJsonAndUsageMetadata() -> None:
    chat_model = FakeChatModel(
        AIMessage(
            content='{"strategy":"direct","reason":"问题明确"}',
            usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        )
    )
    adapter = LangChainRewriteModel(
        make_settings(),
        chat_model=cast(BaseChatModel, cast(Any, chat_model)),
    )

    result = adapter.invoke([HumanMessage(content="AI课程学费是多少？")])

    assert result.strategy is RewriteStrategy.DIRECT
    assert result.input_tokens == 12
    assert result.output_tokens == 3
