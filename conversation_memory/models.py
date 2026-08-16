"""会话记忆领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ConversationSession:
    """一个可恢复的用户会话。"""

    session_id: str
    title: str
    memory_turn_limit: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ConversationTurn:
    """一轮完整的用户问题与助手回答。"""

    turn_number: int
    user_question: str
    assistant_answer: str
    created_at: datetime


@dataclass(frozen=True)
class MemoryContext:
    """本次请求实际选中的历史上下文。"""

    session_id: str
    messages: tuple[ConversationTurn, ...]
    summary: str | None = None

    @property
    def selected_turn_count(self) -> int:
        return len(self.messages)
