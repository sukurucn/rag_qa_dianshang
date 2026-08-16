from __future__ import annotations

import json

import httpx
import pytest

from src.streamlitFrontend.apiClient import QueryApiClient, QueryApiError


def make_client(handler: httpx.MockTransport) -> QueryApiClient:
    return QueryApiClient("http://testserver", client=httpx.Client(transport=handler))


def testSendsQuestionAndParsesCitations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/query"
        assert json.loads(request.content) == {"question": "如何退货？"}
        return httpx.Response(
            200,
            json={
                "answer": "请在订单页申请。",
                "source": "rag",
                "classification": "professional_consultation",
                "faq_confidence": 0.32,
                "classification_confidence": 0.91,
                "fallback_reason": None,
                "citations": [
                    {
                        "chunk_id": "a" * 32,
                        "source": "manual.md",
                        "title_path": "售后 > 退货",
                        "text": "退货说明",
                    }
                ],
            },
        )

    result = make_client(httpx.MockTransport(handler)).ask("  如何退货？  ")

    assert result.answer == "请在订单页申请。"
    assert result.citations[0].title_path == "售后 > 退货"


def testRaisesReadableErrorForUnavailableBackend() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    with pytest.raises(QueryApiError, match="无法连接"):
        make_client(httpx.MockTransport(handler)).ask("测试问题")


def testUsesBackendValidationDetail() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "question must not be blank"})

    with pytest.raises(QueryApiError, match="question must not be blank"):
        make_client(httpx.MockTransport(handler)).ask("测试问题")
