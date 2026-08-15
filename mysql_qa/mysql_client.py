"""MySQL 问答表的连接、建表与数据访问。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import md5
from typing import Any

import pymysql  # type: ignore[import-untyped]

from base.config import Settings
from base.logger import get_logger
from mysql_qa.models import QaWriteResult, QuestionAnswer


class MysqlQaClient:
    """隔离问题答案表的 MySQL 访问，并在关键操作记录日志。"""

    def __init__(self, settings: Settings, connection_factory: Callable[..., Any] = pymysql.connect) -> None:
        self._settings = settings
        self._connection_factory = connection_factory
        self._connection: Any | None = None
        self._logger = get_logger("mysql_qa.mysql_client")

    @staticmethod
    def question_id(question: str) -> str:
        """使用 UTF-8 问题文本的 MD5 作为稳定的主键。"""
        return md5(question.encode("utf-8"), usedforsecurity=False).hexdigest()

    def connect(self) -> None:
        """建立 MySQL 连接；连接失败时记录异常并向调用方报告。"""
        if self._connection is not None:
            return
        self._logger.info(
            "Connecting to MySQL for mysql_qa: host=%s port=%s database=%s",
            self._settings.mysql_host,
            self._settings.mysql_port,
            self._settings.mysql_database,
        )
        try:
            self._connection = self._connection_factory(
                host=self._settings.mysql_host,
                port=self._settings.mysql_port,
                user=self._settings.mysql_user,
                password=self._settings.mysql_password.get_secret_value(),
                database=self._settings.mysql_database,
                charset=self._settings.mysql_charset,
                autocommit=False,
            )
            self._logger.info("MySQL connection for mysql_qa established")
        except Exception:
            self._logger.exception("Unable to connect to MySQL for mysql_qa")
            raise

    def initialize_schema(self) -> None:
        """创建缺失的问题答案表，操作幂等。"""
        connection = self._require_connection()
        table_name = self._settings.mysql_qa_table_name
        statement = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                id CHAR(32) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                PRIMARY KEY (id)
            ) CHARACTER SET {self._settings.mysql_charset}
        """
        self._logger.info("Initializing mysql_qa table: table=%s", table_name)
        try:
            with connection.cursor() as cursor:
                cursor.execute(statement)
            connection.commit()
            self._logger.info("mysql_qa table is ready: table=%s", table_name)
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to initialize mysql_qa table: table=%s", table_name)
            raise

    def upsert(self, question: str, answer: str) -> str:
        """按问题 MD5 幂等写入答案，返回问题 ID。"""
        if not question.strip():
            raise ValueError("question must not be empty")
        if not answer.strip():
            raise ValueError("answer must not be empty")
        connection = self._require_connection()
        question_id = self.question_id(question)
        table_name = self._settings.mysql_qa_table_name
        statement = f"""
            INSERT INTO `{table_name}` (id, question, answer)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE question = VALUES(question), answer = VALUES(answer)
        """
        self._logger.info("Storing mysql_qa answer: question_id=%s", question_id)
        try:
            with connection.cursor() as cursor:
                cursor.execute(statement, (question_id, question, answer))
            connection.commit()
            self._logger.info("Stored mysql_qa answer: question_id=%s", question_id)
            return question_id
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to store mysql_qa answer: question_id=%s", question_id)
            raise

    def bulk_upsert(self, items: Sequence[tuple[str, str]]) -> QaWriteResult:
        """在单个 MySQL 事务内批量写入 QA，并返回新增与覆盖数量。"""
        normalized_items = [(question.strip(), answer.strip()) for question, answer in items]
        if not normalized_items:
            return QaWriteResult(question_ids=(), created_count=0, updated_count=0)
        if any(not question or not answer for question, answer in normalized_items):
            raise ValueError("question and answer must not be empty")

        connection = self._require_connection()
        table_name = self._settings.mysql_qa_table_name
        question_ids = tuple(self.question_id(question) for question, _ in normalized_items)
        placeholders = ", ".join(["%s"] * len(question_ids))
        select_statement = f"SELECT id FROM `{table_name}` WHERE id IN ({placeholders})"
        upsert_statement = f"""
            INSERT INTO `{table_name}` (id, question, answer)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE question = VALUES(question), answer = VALUES(answer)
        """
        try:
            with connection.cursor() as cursor:
                cursor.execute(select_statement, question_ids)
                existing_ids = {row[0] for row in cursor.fetchall()}
                for question_id, (question, answer) in zip(question_ids, normalized_items, strict=True):
                    cursor.execute(upsert_statement, (question_id, question, answer))
            connection.commit()
            updated_count = sum(question_id in existing_ids for question_id in question_ids)
            self._logger.info("Stored mysql_qa batch: count=%s", len(question_ids))
            return QaWriteResult(
                question_ids=question_ids,
                created_count=len(question_ids) - updated_count,
                updated_count=updated_count,
            )
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to store mysql_qa batch: count=%s", len(question_ids))
            raise

    def list_question_answers(self, *, limit: int, offset: int) -> list[QuestionAnswer]:
        """分页读取完整 QA 记录，供本地管理 API 使用。"""
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset must not be negative")
        connection = self._require_connection()
        table_name = self._settings.mysql_qa_table_name
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT id, question, answer FROM `{table_name}` ORDER BY id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                rows: Sequence[tuple[str, str, str]] = cursor.fetchall()
            return [QuestionAnswer(question_id=row[0], question=row[1], answer=row[2]) for row in rows]
        except Exception:
            self._logger.exception("Failed to list mysql_qa records")
            raise

    def delete_many(self, question_ids: Sequence[str]) -> int:
        """在一个事务中按问题主键批量删除 QA，返回实际删除数量。"""
        if not question_ids:
            return 0
        if any(len(question_id) != 32 for question_id in question_ids):
            raise ValueError("question_ids must be MD5 hashes")
        connection = self._require_connection()
        table_name = self._settings.mysql_qa_table_name
        placeholders = ", ".join(["%s"] * len(question_ids))
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM `{table_name}` WHERE id IN ({placeholders})", tuple(question_ids))
                deleted_count = int(cursor.rowcount)
            connection.commit()
            self._logger.info("Deleted mysql_qa records: count=%s", deleted_count)
            return deleted_count
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to delete mysql_qa records")
            raise

    def list_questions(self) -> list[QuestionAnswer]:
        """读取全部问题，供 Redis warmup 使用。"""
        connection = self._require_connection()
        table_name = self._settings.mysql_qa_table_name
        self._logger.info("Loading mysql_qa questions for Redis warmup: table=%s", table_name)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT id, question FROM `{table_name}`")
                rows: Sequence[tuple[str, str]] = cursor.fetchall()
            questions = [QuestionAnswer(question_id=row[0], question=row[1]) for row in rows]
            self._logger.info("Loaded mysql_qa questions: count=%s", len(questions))
            return questions
        except Exception:
            self._logger.exception("Failed to load mysql_qa questions")
            raise

    def get_answer(self, question_id: str) -> QuestionAnswer | None:
        """按稳定的问题 ID 读取问题和答案。"""
        connection = self._require_connection()
        table_name = self._settings.mysql_qa_table_name
        self._logger.info("Looking up mysql_qa answer: question_id=%s", question_id)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT id, question, answer FROM `{table_name}` WHERE id = %s", question_id
                )
                row: tuple[str, str, str] | None = cursor.fetchone()
            if row is None:
                self._logger.warning("mysql_qa answer not found: question_id=%s", question_id)
                return None
            return QuestionAnswer(question_id=row[0], question=row[1], answer=row[2])
        except Exception:
            self._logger.exception("Failed to retrieve mysql_qa answer: question_id=%s", question_id)
            raise

    def close(self) -> None:
        """关闭已建立的 MySQL 连接。"""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._logger.info("MySQL connection for mysql_qa closed")

    def _require_connection(self) -> Any:
        """返回可用连接，供内部数据库操作复用。"""
        self.connect()
        assert self._connection is not None
        return self._connection
