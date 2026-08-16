from __future__ import annotations

from typing import Any, cast

from fastapi.testclient import TestClient

from query_api.app import QueryApiServices, create_app
from query_api.feature_flags import FeatureFlags
from query_api.models import QueryAnswer


class FakeAnswerService:
    def answer(self, question: str) -> QueryAnswer:
        return QueryAnswer("FAQ 答案", "faq", None, (), 0.98)


class FakeServices:
    def __init__(self) -> None:
        self.answer_service = FakeAnswerService()
        self.feature_flags = FakeFeatureFlags()
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeFeatureFlags:
    def __init__(self) -> None:
        self.flags = FeatureFlags()

    def current(self) -> FeatureFlags:
        return self.flags

    def update(self, *, faq_enabled: bool | None, classifier_enabled: bool | None) -> FeatureFlags:
        self.flags = FeatureFlags(
            faq_enabled=self.flags.faq_enabled if faq_enabled is None else faq_enabled,
            classifier_enabled=self.flags.classifier_enabled if classifier_enabled is None else classifier_enabled,
        )
        return self.flags


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


def testFeatureEndpointsReturnAndPersistStates() -> None:
    app = create_app(services=cast(QueryApiServices, cast(Any, FakeServices())))

    with TestClient(app) as client:
        initial = client.get("/runtime/features")
        updated = client.patch("/runtime/features", json={"faq_enabled": False})

    assert initial.json() == {"faq_enabled": True, "classifier_enabled": True}
    assert updated.json() == {"faq_enabled": False, "classifier_enabled": True}
