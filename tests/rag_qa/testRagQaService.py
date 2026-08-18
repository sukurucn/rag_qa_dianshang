from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from question_rewrite.models import RewriteResult
from rag_qa.answer_client import LangChainAnswerModel
from rag_qa.models import AnswerGeneration, ParentChunk, RetrievalReport, WebSearchResult
from rag_qa.service import RagQaService
from tests.mysql_qa.testMysqlClient import make_settings


class FakeRetriever:
    def __init__(self, report: RetrievalReport | Exception) -> None:
        self._report = report
        self.queries: tuple[str, ...] = ()

    def retrieve(self, queries: Sequence[str]) -> RetrievalReport:
        self.queries = tuple(queries)
        if isinstance(self._report, Exception):
            raise self._report
        return self._report


class FakeReranker:
    def __init__(self, ranked: list[ParentChunk]) -> None:
        self._ranked = ranked
        self.queries: tuple[str, ...] = ()
        self.was_called = False

    def rerank(self, queries: Sequence[str], parents: Sequence[ParentChunk]) -> list[ParentChunk]:
        self.was_called = True
        self.queries = tuple(queries)
        return self._ranked


class FakeAnswerModel:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, tuple[ParentChunk, ...]]] = []

    def answer(
        self,
        question: str,
        parents: Sequence[ParentChunk],
    ) -> AnswerGeneration:
        self.calls.append((question, tuple(parents)))
        return AnswerGeneration(self._response)


class FakeChatModel:
    def __init__(self, response: str | list[AIMessage]) -> None:
        self._response = response
        self.messages: list[BaseMessage] = []

    def bind_tools(self, tools: list[object]) -> FakeChatModel:
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.messages = messages
        if isinstance(self._response, list):
            return self._response.pop(0)
        return AIMessage(content=self._response)


def parent(identifier: str) -> ParentChunk:
    return ParentChunk(chunk_id=identifier, source=f"{identifier}.md", title_path="课程", text=f"原文 {identifier}")


def rewrite_result() -> RewriteResult:
    return RewriteResult(
        original_question="AI课程收费是多少？",
        rag_queries=("AI课程收费是多少？", "AI课程费用说明"),
        strategy_traces=(),
        max_context_tokens=0,
        total_input_tokens=0,
        total_output_tokens=0,
    )


def make_service(
    report: RetrievalReport | Exception,
    reranked: list[ParentChunk],
    answer: str = "课程费用见报价单。",
) -> tuple[RagQaService, FakeReranker, FakeAnswerModel]:
    reranker = FakeReranker(reranked)
    answer_model = FakeAnswerModel(answer)
    service = RagQaService(make_settings(), FakeRetriever(report), reranker, answer_model)  # type: ignore[arg-type]
    return service, reranker, answer_model


def testLetsAnswerAgentHandleEmptyMilvusContext() -> None:
    service, reranker, answer_model = make_service(RetrievalReport((), 0, 0), [])

    result = service.answer(rewrite_result())

    assert result.answer == "课程费用见报价单。"
    assert result.fallback_reason is None
    assert reranker.was_called is False
    assert answer_model.calls[0][1] == ()


def testAnswersDirectlyWithSingleParent() -> None:
    only_parent = parent("one")
    service, reranker, answer_model = make_service(RetrievalReport((only_parent,), 5, 5), [])

    result = service.answer(rewrite_result())

    assert result.answer == "课程费用见报价单。"
    assert result.parents == (only_parent,)
    assert reranker.was_called is False
    assert answer_model.calls[0][1] == (only_parent,)


def testAnswersWithEmptyMilvusContextLetsAgentUseTools() -> None:
    service, _, answer_model = make_service(RetrievalReport((), 0, 0), [])
    result = service.answer(rewrite_result())

    assert result.fallback_reason is None
    assert result.web_results == ()
    assert answer_model.calls[0][1] == ()


def testReranksMultipleParentsUsingAllRewriteAndHydeQueries() -> None:
    parents = (parent("one"), parent("two"), parent("three"))
    service, reranker, answer_model = make_service(
        RetrievalReport(parents, 5, 5),
        [parents[2], parents[1], parents[0]],
    )

    result = service.answer(rewrite_result())

    assert reranker.was_called is True
    assert reranker.queries == ("AI课程收费是多少？", "AI课程费用说明")
    assert result.parents == (parents[2], parents[1])
    assert answer_model.calls[0][1] == (parents[2], parents[1])


def testReturnsCustomerServiceWhenAnswerIsNotGrounded() -> None:
    service, _, _ = make_service(RetrievalReport((parent("one"),), 5, 5), [], answer="UNANSWERABLE")

    result = service.answer(rewrite_result())

    assert result.answer == "客服电话：30129032"
    assert result.fallback_reason == "answer_not_grounded"


def testReturnsCustomerServiceWhenRetrievalFails() -> None:
    service, _, _ = make_service(RuntimeError("Milvus unavailable"), [])

    result = service.answer(rewrite_result())

    assert result.answer == "客服电话：30129032"
    assert result.fallback_reason == "rag_qa_error"


def testAnswerPromptContainsOriginalParentTextAndSourceMetadata() -> None:
    chat_model = FakeChatModel("费用请参考课程报价。[来源：course.md / 费用]")
    answer_model = LangChainAnswerModel(
        make_settings(),
        chat_model=cast(BaseChatModel, cast(Any, chat_model)),
    )

    answer = answer_model.answer("学费是多少？", [ParentChunk("parent-1", "course.md", "费用", "课程费用为 100 元")])

    assert answer.text.startswith("费用请参考")
    prompt = str(chat_model.messages[1].content)
    assert "来源：course.md" in prompt
    assert "章节：费用" in prompt
    assert "原文：课程费用为 100 元" in prompt
    assert "本地 RAG 原文（优先级最高）" in prompt


def testAnswerPromptLetsAgentUseWebOnlyAsSupplement() -> None:
    chat_model = FakeChatModel("网络补充答案")
    answer_model = LangChainAnswerModel(
        make_settings(),
        chat_model=cast(BaseChatModel, cast(Any, chat_model)),
    )

    answer_model.answer("Python 是什么？", [ParentChunk("parent-1", "course.md", "费用", "课程费用为 100 元")])

    system_prompt = str(chat_model.messages[0].content)
    user_prompt = str(chat_model.messages[1].content)
    assert "本地知识库（RAG）原文是最高优先级事实依据" in system_prompt
    assert "search_web 工具" in system_prompt
    assert "自行决定是否调用" in system_prompt
    assert "本地 RAG 原文（优先级最高）" in user_prompt


def testAnswerAgentInvokesWebToolOnlyWhenModelRequestsIt() -> None:
    tool_request = AIMessage(
        content="",
        tool_calls=[{"name": "search_web", "args": {"query": "Python"}, "id": "call-1"}],
    )
    chat_model = FakeChatModel([tool_request, AIMessage(content="Python 是一种编程语言。")])

    class FakeWebSearcher:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str) -> tuple[WebSearchResult, ...]:
            self.queries.append(query)
            return (WebSearchResult("Python", "https://example.test/python", "编程语言"),)

    web_searcher = FakeWebSearcher()
    answer_model = LangChainAnswerModel(
        make_settings(),
        chat_model=cast(BaseChatModel, cast(Any, chat_model)),
        web_searcher=web_searcher,  # type: ignore[arg-type]
    )

    answer = answer_model.answer("Python 是什么？", ())

    assert web_searcher.queries == ["Python"]
    assert answer.text == "Python 是一种编程语言。"
    assert answer.web_results[0].url == "https://example.test/python"
