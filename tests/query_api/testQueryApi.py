from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

from fastapi.testclient import TestClient

from conversation_memory.models import ConversationSession, MemoryContext
from query_api.app import QueryApiServices, create_app
from query_api.feature_flags import FeatureFlags
from query_api.models import QueryAnswer


class FakeAnswerService:
    def answer(self, question: str, history: tuple[object, ...] = ()) -> QueryAnswer:
        return QueryAnswer("FAQ 答案", "faq", None, (), 0.98)


class FakeServices:
    def __init__(self) -> None:
        self.answer_service = FakeAnswerService()
        self.feature_flags = FakeFeatureFlags()
        self.conversations = FakeConversations()
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


class FakeConversations:
    def __init__(self) -> None:
        now = datetime(2026, 8, 16, tzinfo=timezone.utc)
        self.session = ConversationSession("a" * 36, "新会话", 12, now, now)

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def resolve(self, session_id: str | None) -> ConversationSession:
        if session_id is not None and session_id != self.session.session_id:
            raise KeyError(session_id)
        return self.session

    def select_context(self, session_id: str, question: str) -> MemoryContext:
        return MemoryContext(session_id=session_id, messages=())

    def to_messages(self, context: MemoryContext) -> tuple[object, ...]:
        return ()

    def append(self, session_id: str, question: str, answer: str) -> int:
        return 1

    def create(self, *, title: str | None, memory_turn_limit: int) -> ConversationSession:
        return self.session

    def list_sessions(self, *, limit: int, offset: int) -> list[ConversationSession]:
        return [self.session]

    def update(
        self, session_id: str, *, title: str | None, memory_turn_limit: int | None
    ) -> ConversationSession:
        return self.session

    def get_turns(self, session_id: str, *, limit: int, offset: int) -> list[object]:
        return []

    def delete(self, session_id: str) -> None:
        return None

def testQueryEndpointReturnsUnifiedResponse() -> None:
    services = FakeServices()
    app = create_app(services=cast(QueryApiServices, cast(Any, services)))

    with TestClient(app) as client:
        response = client.post("/query", json={"question": "如何退款？"})

    assert response.status_code == 200
    assert response.json()["source"] == "faq"
    assert response.json()["citations"] == []
    assert response.json()["session_id"] == "a" * 36
    assert response.json()["turn_number"] == 1
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


def testSessionEndpointsExposePersistentConversationContract() -> None:
    app = create_app(services=cast(QueryApiServices, cast(Any, FakeServices())))

    with TestClient(app) as client:
        created = client.post("/sessions", json={"memory_turn_limit": 24})
        sessions = client.get("/sessions")
        turns = client.get(f"/sessions/{'a' * 36}/turns")

    assert created.status_code == 201
    assert created.json()["memory_turn_limit"] == 12
    assert sessions.json()[0]["session_id"] == "a" * 36
    assert turns.json() == []
