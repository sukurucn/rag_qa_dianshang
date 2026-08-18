"""会话记忆的 MySQL 持久化边界。"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Protocol

import pymysql  # type: ignore[import-untyped]

from base.config import Settings
from base.logger import get_logger
from conversation_memory.models import ConversationSession, ConversationTurn

MAX_MEMORY_TURNS = 256


class ConversationStore(Protocol):
    """会话存储所需的最小接口，便于用替身测试。"""

    def initialize_schema(self) -> None: ...

    def create(self, *, memory_turn_limit: int, title: str | None = None) -> ConversationSession: ...

    def get(self, session_id: str) -> ConversationSession | None: ...

    def list_sessions(self, *, limit: int, offset: int) -> list[ConversationSession]: ...

    def update(self, session_id: str, *, title: str | None, memory_turn_limit: int | None) -> ConversationSession: ...

    def delete(self, session_id: str) -> None: ...

    def list_turns(self, session_id: str, *, limit: int, offset: int) -> list[ConversationTurn]: ...

    def append_turn(self, session_id: str, question: str, answer: str) -> tuple[int, list[ConversationTurn]]: ...

    def get_summary(self, session_id: str) -> str | None: ...

    def save_summary(self, session_id: str, summary: str) -> None: ...

    def close(self) -> None: ...


class MysqlConversationStore:
    """将会话、轮次和压缩摘要持久化到独立 MySQL 表。"""

    _SESSIONS_TABLE = "chat_sessions"
    _TURNS_TABLE = "chat_turns"
    _SUMMARIES_TABLE = "chat_memory_summaries"

    def __init__(self, settings: Settings, connection_factory: Callable[..., Any] = pymysql.connect) -> None:
        self._settings = settings
        self._connection_factory = connection_factory
        self._connection: Any | None = None
        self._logger = get_logger("conversation_memory.repository")

    def initialize_schema(self) -> None:
        connection = self._require_connection()
        statements = (
            f"""
            CREATE TABLE IF NOT EXISTS `{self._SESSIONS_TABLE}` (
                session_id CHAR(36) NOT NULL,
                title VARCHAR(128) NOT NULL,
                memory_turn_limit SMALLINT UNSIGNED NOT NULL,
                created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                    ON UPDATE CURRENT_TIMESTAMP(6),
                PRIMARY KEY (session_id)
            ) CHARACTER SET {self._settings.mysql_charset}
            """,
            f"""
            CREATE TABLE IF NOT EXISTS `{self._TURNS_TABLE}` (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                session_id CHAR(36) NOT NULL,
                turn_number SMALLINT UNSIGNED NOT NULL,
                user_question MEDIUMTEXT NOT NULL,
                assistant_answer MEDIUMTEXT NOT NULL,
                created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                PRIMARY KEY (id),
                UNIQUE KEY `uq_chat_turn_session_number` (session_id, turn_number),
                KEY `idx_chat_turn_session_created` (session_id, created_at)
            ) CHARACTER SET {self._settings.mysql_charset}
            """,
            f"""
            CREATE TABLE IF NOT EXISTS `{self._SUMMARIES_TABLE}` (
                session_id CHAR(36) NOT NULL,
                summary MEDIUMTEXT NOT NULL,
                updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                    ON UPDATE CURRENT_TIMESTAMP(6),
                PRIMARY KEY (session_id)
            ) CHARACTER SET {self._settings.mysql_charset}
            """,
        )
        try:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to initialize conversation-memory tables")
            raise

    def create(self, *, memory_turn_limit: int, title: str | None = None) -> ConversationSession:
        self._validate_turn_limit(memory_turn_limit)
        session_id = str(uuid.uuid4())
        normalized_title = (title or "新会话").strip()[:128] or "新会话"
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO `{self._SESSIONS_TABLE}` (session_id, title, memory_turn_limit) "
                    "VALUES (%s, %s, %s)",
                    (session_id, normalized_title, memory_turn_limit),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to create conversation session")
            raise
        session = self.get(session_id)
        assert session is not None
        return session

    def get(self, session_id: str) -> ConversationSession | None:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT session_id, title, memory_turn_limit, created_at, updated_at "
                    f"FROM `{self._SESSIONS_TABLE}` WHERE session_id = %s",
                    (session_id,),
                )
                row = cursor.fetchone()
            connection.rollback()
            return self._session_from_row(row) if row is not None else None
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to read conversation session")
            raise

    def list_sessions(self, *, limit: int, offset: int) -> list[ConversationSession]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset must not be negative")
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT session_id, title, memory_turn_limit, created_at, updated_at "
                    f"FROM `{self._SESSIONS_TABLE}` ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows = cursor.fetchall()
            connection.rollback()
            return [self._session_from_row(row) for row in rows]
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to list conversation sessions")
            raise

    def update(self, session_id: str, *, title: str | None, memory_turn_limit: int | None) -> ConversationSession:
        if title is None and memory_turn_limit is None:
            raise ValueError("at least one session field must be supplied")
        if memory_turn_limit is not None:
            self._validate_turn_limit(memory_turn_limit)
        assignments: list[str] = []
        params: list[object] = []
        if title is not None:
            assignments.append("title = %s")
            params.append(title.strip()[:128] or "新会话")
        if memory_turn_limit is not None:
            assignments.append("memory_turn_limit = %s")
            params.append(memory_turn_limit)
        params.append(session_id)
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE `{self._SESSIONS_TABLE}` SET {', '.join(assignments)} WHERE session_id = %s",
                    tuple(params),
                )
                if cursor.rowcount == 0:
                    raise KeyError(session_id)
                if memory_turn_limit is not None:
                    cursor.execute(
                        f"SELECT COALESCE(MAX(turn_number), 0) FROM `{self._TURNS_TABLE}` "
                        "WHERE session_id = %s",
                        (session_id,),
                    )
                    latest_turn = int(cursor.fetchone()[0])
                    cutoff = latest_turn - memory_turn_limit
                    if cutoff > 0:
                        cursor.execute(
                            f"DELETE FROM `{self._TURNS_TABLE}` "
                            "WHERE session_id = %s AND turn_number <= %s",
                            (session_id, cutoff),
                        )
                    if memory_turn_limit == 0:
                        cursor.execute(f"DELETE FROM `{self._TURNS_TABLE}` WHERE session_id = %s", (session_id,))
                        cursor.execute(f"DELETE FROM `{self._SUMMARIES_TABLE}` WHERE session_id = %s", (session_id,))
            connection.commit()
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to update conversation session")
            raise
        session = self.get(session_id)
        assert session is not None
        return session

    def delete(self, session_id: str) -> None:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM `{self._SUMMARIES_TABLE}` WHERE session_id = %s", (session_id,))
                cursor.execute(f"DELETE FROM `{self._TURNS_TABLE}` WHERE session_id = %s", (session_id,))
                cursor.execute(f"DELETE FROM `{self._SESSIONS_TABLE}` WHERE session_id = %s", (session_id,))
                if cursor.rowcount == 0:
                    raise KeyError(session_id)
            connection.commit()
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to delete conversation session")
            raise

    def list_turns(self, session_id: str, *, limit: int, offset: int) -> list[ConversationTurn]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset must not be negative")
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT turn_number, user_question, assistant_answer, created_at "
                    f"FROM `{self._TURNS_TABLE}` WHERE session_id = %s "
                    "ORDER BY turn_number DESC LIMIT %s OFFSET %s",
                    (session_id, limit, offset),
                )
                rows = cursor.fetchall()
            connection.rollback()
            return [self._turn_from_row(row) for row in reversed(rows)]
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to list conversation turns")
            raise

    def append_turn(self, session_id: str, question: str, answer: str) -> tuple[int, list[ConversationTurn]]:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT memory_turn_limit FROM `{self._SESSIONS_TABLE}` "
                    "WHERE session_id = %s FOR UPDATE",
                    (session_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(session_id)
                memory_turn_limit = int(row[0])
                if memory_turn_limit == 0:
                    connection.commit()
                    return 0, []
                cursor.execute(
                    f"SELECT COALESCE(MAX(turn_number), 0) FROM `{self._TURNS_TABLE}` WHERE session_id = %s",
                    (session_id,),
                )
                turn_number = int(cursor.fetchone()[0]) + 1
                cursor.execute(
                    f"INSERT INTO `{self._TURNS_TABLE}` "
                    "(session_id, turn_number, user_question, assistant_answer) VALUES (%s, %s, %s, %s)",
                    (session_id, turn_number, question, answer),
                )
                cutoff = turn_number - memory_turn_limit
                evicted: list[ConversationTurn] = []
                if cutoff > 0:
                    cursor.execute(
                        f"SELECT turn_number, user_question, assistant_answer, created_at "
                        f"FROM `{self._TURNS_TABLE}` WHERE session_id = %s AND turn_number <= %s "
                        "ORDER BY turn_number",
                        (session_id, cutoff),
                    )
                    evicted = [self._turn_from_row(item) for item in cursor.fetchall()]
                    cursor.execute(
                        f"DELETE FROM `{self._TURNS_TABLE}` WHERE session_id = %s AND turn_number <= %s",
                        (session_id, cutoff),
                    )
                cursor.execute(
                    f"UPDATE `{self._SESSIONS_TABLE}` SET updated_at = CURRENT_TIMESTAMP(6) WHERE session_id = %s",
                    (session_id,),
                )
            connection.commit()
            return turn_number, evicted
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to append conversation turn")
            raise

    def get_summary(self, session_id: str) -> str | None:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT summary FROM `{self._SUMMARIES_TABLE}` WHERE session_id = %s", (session_id,)
                )
                row = cursor.fetchone()
            connection.rollback()
            return str(row[0]) if row is not None else None
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to read conversation summary")
            raise

    def save_summary(self, session_id: str, summary: str) -> None:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"INSERT INTO `{self._SUMMARIES_TABLE}` (session_id, summary) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE summary = VALUES(summary)",
                    (session_id, summary),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to save conversation summary")
            raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _require_connection(self) -> Any:
        if self._connection is None:
            self._connection = self._connection_factory(
                host=self._settings.mysql_host,
                port=self._settings.mysql_port,
                user=self._settings.mysql_user,
                password=self._settings.mysql_password.get_secret_value(),
                database=self._settings.mysql_database,
                charset=self._settings.mysql_charset,
                autocommit=False,
            )
        return self._connection

    @staticmethod
    def _validate_turn_limit(memory_turn_limit: int) -> None:
        if not 0 <= memory_turn_limit <= MAX_MEMORY_TURNS:
            raise ValueError(f"memory_turn_limit must be between 0 and {MAX_MEMORY_TURNS}")

    @staticmethod
    def _session_from_row(row: Sequence[object]) -> ConversationSession:
        return ConversationSession(
            session_id=str(row[0]),
            title=str(row[1]),
            memory_turn_limit=int(row[2]),
            created_at=row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(str(row[3])),
            updated_at=row[4] if isinstance(row[4], datetime) else datetime.fromisoformat(str(row[4])),
        )

    @staticmethod
    def _turn_from_row(row: Sequence[object]) -> ConversationTurn:
        created_at = row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(str(row[3]))
        return ConversationTurn(int(row[0]), str(row[1]), str(row[2]), created_at)
