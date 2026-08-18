from __future__ import annotations

from datetime import datetime, timezone

from conversation_memory.models import ConversationSession, ConversationTurn
from conversation_memory.service import ConversationMemoryService


class InMemoryStore:
    def __init__(self) -> None:
        now = datetime(2026, 8, 16, tzinfo=timezone.utc)
        self.sessions: dict[str, ConversationSession] = {
            "session-1": ConversationSession("session-1", "新会话", 2, now, now),
            "session-0": ConversationSession("session-0", "无历史", 0, now, now),
        }
        self.turns: dict[str, list[ConversationTurn]] = {"session-1": [], "session-0": []}
        self.summaries: dict[str, str] = {}

    def initialize_schema(self) -> None: pass
    def close(self) -> None: pass
    def create(self, *, memory_turn_limit: int, title: str | None = None) -> ConversationSession: return self.sessions["session-1"]
    def get(self, session_id: str) -> ConversationSession | None: return self.sessions.get(session_id)
    def list_sessions(self, *, limit: int, offset: int) -> list[ConversationSession]: return list(self.sessions.values())
    def update(self, session_id: str, *, title: str | None, memory_turn_limit: int | None) -> ConversationSession: return self.sessions[session_id]
    def delete(self, session_id: str) -> None: self.sessions.pop(session_id)
    def get_summary(self, session_id: str) -> str | None: return self.summaries.get(session_id)
    def save_summary(self, session_id: str, summary: str) -> None: self.summaries[session_id] = summary

    def list_turns(self, session_id: str, *, limit: int, offset: int) -> list[ConversationTurn]:
        return self.turns[session_id][offset : offset + limit]

    def append_turn(self, session_id: str, question: str, answer: str) -> tuple[int, list[ConversationTurn]]:
        session = self.sessions[session_id]
        if session.memory_turn_limit == 0:
            return 0, []
        items = self.turns[session_id]
        number = items[-1].turn_number + 1 if items else 1
        items.append(ConversationTurn(number, question, answer, datetime(2026, 8, 16, tzinfo=timezone.utc)))
        evicted = items[:-session.memory_turn_limit]
        self.turns[session_id] = items[-session.memory_turn_limit :]
        return number, evicted


def testKeepsConfiguredTurnLimitAndSummarizesEvictedTurns() -> None:
    store = InMemoryStore()
    service = ConversationMemoryService(store)  # type: ignore[arg-type]

    service.append("session-1", "我想了解退款流程", "可以在订单页申请退款。")
    service.append("session-1", "退款多久到账", "通常会在三到五个工作日到账。")
    service.append("session-1", "退款需要什么材料", "请准备订单信息。")

    assert [item.turn_number for item in store.turns["session-1"]] == [2, 3]
    assert "我想了解退款流程" in store.summaries["session-1"]


def testSelectsRelevantTurnsAndDoesNotPersistZeroMemorySession() -> None:
    store = InMemoryStore()
    service = ConversationMemoryService(store)  # type: ignore[arg-type]
    service.append("session-1", "我想了解退款流程", "可以在订单页申请退款。")
    service.append("session-1", "物流状态在哪里看", "可以在订单详情中查看物流。")

    context = service.select_context("session-1", "退款到账时间是什么时候")
    no_memory_turn = service.append("session-0", "这条消息不保存", "不会被保存")

    assert context.selected_turn_count == 2
    assert context.messages[0].user_question == "我想了解退款流程"
    assert no_memory_turn == 0
    assert store.turns["session-0"] == []
