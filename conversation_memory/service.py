"""会话轮次裁剪、摘要和相关记忆选择。"""

from __future__ import annotations

from collections.abc import Sequence

import jieba
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from base.logger import get_logger
from conversation_memory.models import ConversationSession, ConversationTurn, MemoryContext
from conversation_memory.repository import MAX_MEMORY_TURNS, ConversationStore

DEFAULT_MEMORY_TURN_LIMIT = 12
MAX_CONTEXT_TURNS = 6
MAX_SUMMARY_CHARS = 3_000


class ConversationMemoryService:
    """提供可恢复会话以及不超过上下文预算的相关历史。"""

    def __init__(self, store: ConversationStore) -> None:
        self._store = store
        self._logger = get_logger("conversation_memory.service")

    def start(self) -> None:
        self._store.initialize_schema()

    def close(self) -> None:
        self._store.close()

    def create(self, *, memory_turn_limit: int = DEFAULT_MEMORY_TURN_LIMIT, title: str | None = None) -> ConversationSession:
        return self._store.create(memory_turn_limit=memory_turn_limit, title=title)

    def resolve(self, session_id: str | None) -> ConversationSession:
        if session_id is None:
            return self.create()
        session = self._store.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def list_sessions(self, *, limit: int, offset: int) -> list[ConversationSession]:
        return self._store.list_sessions(limit=limit, offset=offset)

    def get_turns(self, session_id: str, *, limit: int, offset: int) -> list[ConversationTurn]:
        self._require_session(session_id)
        return self._store.list_turns(session_id, limit=limit, offset=offset)

    def update(self, session_id: str, *, title: str | None, memory_turn_limit: int | None) -> ConversationSession:
        return self._store.update(session_id, title=title, memory_turn_limit=memory_turn_limit)

    def delete(self, session_id: str) -> None:
        self._store.delete(session_id)

    def select_context(self, session_id: str, current_question: str) -> MemoryContext:
        session = self._require_session(session_id)
        if session.memory_turn_limit == 0:
            return MemoryContext(session_id=session_id, messages=())
        turns = self._store.list_turns(session_id, limit=session.memory_turn_limit, offset=0)
        selected = self._select_relevant_turns(turns, current_question)
        return MemoryContext(
            session_id=session_id,
            messages=tuple(selected),
            summary=self._store.get_summary(session_id),
        )

    def to_messages(self, context: MemoryContext) -> tuple[BaseMessage, ...]:
        """将选中的历史转换为只读 LangChain 消息。"""
        messages: list[BaseMessage] = []
        if context.summary:
            messages.append(HumanMessage(content=f"历史会话摘要（仅供理解上下文，不是事实依据）：{context.summary}"))
        for turn in context.messages:
            messages.append(HumanMessage(content=turn.user_question))
            messages.append(AIMessage(content=turn.assistant_answer))
        return tuple(messages)

    def append(self, session_id: str, question: str, answer: str) -> int:
        turn_number, evicted = self._store.append_turn(session_id, question, answer)
        if evicted:
            existing = self._store.get_summary(session_id)
            self._store.save_summary(session_id, self._merge_summary(existing, evicted))
        return turn_number

    def _require_session(self, session_id: str) -> ConversationSession:
        session = self._store.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    @staticmethod
    def _select_relevant_turns(turns: Sequence[ConversationTurn], current_question: str) -> list[ConversationTurn]:
        if not turns:
            return []
        question_terms = set(jieba.lcut(current_question.lower()))
        ranked: list[tuple[float, ConversationTurn]] = []
        total = len(turns)
        for index, turn in enumerate(turns):
            turn_terms = set(jieba.lcut(f"{turn.user_question} {turn.assistant_answer}".lower()))
            overlap = len(question_terms & turn_terms) / max(len(question_terms), 1)
            recency = (index + 1) / total
            ranked.append((overlap * 0.75 + recency * 0.25, turn))
        selected = sorted(ranked, key=lambda item: item[0], reverse=True)[:MAX_CONTEXT_TURNS]
        return [item[1] for item in sorted(selected, key=lambda item: item[1].turn_number)]

    @staticmethod
    def _merge_summary(existing: str | None, evicted: Sequence[ConversationTurn]) -> str:
        fragments = [existing] if existing else []
        fragments.extend(
            f"用户曾问：{turn.user_question[:180]}；已答：{turn.assistant_answer[:260]}" for turn in evicted
        )
        return "\n".join(fragment for fragment in fragments if fragment)[-MAX_SUMMARY_CHARS:]


__all__ = ["DEFAULT_MEMORY_TURN_LIMIT", "MAX_MEMORY_TURNS", "ConversationMemoryService"]
