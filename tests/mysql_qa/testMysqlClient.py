from __future__ import annotations

from typing import Any

from typing_extensions import Self

from base.config import Settings
from mysql_qa.mysql_client import MysqlQaClient


class FakeCursor:
    def __init__(self, rows: list[tuple[str, ...]] | None = None) -> None:
        self.executions: list[tuple[str, tuple[Any, ...] | None]] = []
        self._rows = rows or []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: str, params: tuple[Any, ...] | None = None) -> None:
        self.executions.append((statement, params))

    def fetchall(self) -> list[tuple[str, ...]]:
        return self._rows

    def fetchone(self) -> tuple[str, ...] | None:
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, rows: list[tuple[str, ...]] | None = None) -> None:
        self.cursor_instance = FakeCursor(rows)
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        mysql_host="localhost",
        mysql_user="user",
        mysql_password="password",
        mysql_database="rag_db",
        redis_host="localhost",
        redis_password="password",
        milvus_host="localhost",
        milvus_user="root",
        milvus_password="password",
    )


def testInitializesSchemaAndUpsertsMd5Question() -> None:
    connection = FakeConnection()
    client = MysqlQaClient(make_settings(), connection_factory=lambda **_: connection)

    client.initialize_schema()
    question_id = client.upsert("退货流程是什么？", "请在订单页申请退货。")

    assert question_id == "8a61cbad36e7bbe65853cbf08c06a393"
    assert "CREATE TABLE IF NOT EXISTS `question_answers`" in connection.cursor_instance.executions[0][0]
    assert connection.cursor_instance.executions[1][1] == (question_id, "退货流程是什么？", "请在订单页申请退货。")
    assert connection.commit_count == 2


def testLoadsQuestionsAndAnswer() -> None:
    connection = FakeConnection([("id-1", "如何退款？", "请在订单页申请退款。")])
    client = MysqlQaClient(make_settings(), connection_factory=lambda **_: connection)

    assert client.list_questions()[0].question == "如何退款？"
    assert client.get_answer("id-1").answer == "请在订单页申请退款。"


def testReadOperationsResetTransactionSnapshot() -> None:
    connection = FakeConnection([("id-1", "退款流程是什么？", "请在订单页申请退款。")])
    client = MysqlQaClient(make_settings(), connection_factory=lambda **_: connection)

    client.list_questions()
    client.get_answer("id-1")

    assert connection.rollback_count == 2
    assert connection.cursor_instance.executions[-1][1] == ("id-1",)
