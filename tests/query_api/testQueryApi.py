from __future__ import annotations

from typing import Any, cast

from fastapi.testclient import TestClient

from query_api.app import QueryApiServices, create_app
from query_api.models import QueryAnswer


class FakeAnswerService:
    def answer(self, question: str) -> QueryAnswer:
        return QueryAnswer("FAQ 答案", "faq", None, (), 0.98)


class FakeServices:
    def __init__(self) -> None:
        self.answer_service = FakeAnswerService()
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def testQueryEndpointReturnsUnifiedResponse() -> None:
    services = FakeServices()
    app = create_app(services=cast(QueryApiServices, cast(Any, services)))

    with TestClient(app) as client:
        response = client.post("/query", json={"question": "如何退款？"})

    assert response.status_code == 200
    assert response.json()["source"] == "faq"
    assert response.json()["citations"] == []
    assert services.started is True
    assert services.stopped is True


def testQueryEndpointRejectsBlankQuestion() -> None:
    app = create_app(services=cast(QueryApiServices, cast(Any, FakeServices())))

    with TestClient(app) as client:
        response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422
