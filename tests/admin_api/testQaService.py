from __future__ import annotations

import pytest

from admin_api.qa_service import CacheSynchronizationError, QaAdminService
from mysql_qa.models import QaWriteResult


class FakeMysql:
    def __init__(self) -> None:
        self.items: tuple[tuple[str, str], ...] = ()

    def bulk_upsert(self, items: tuple[tuple[str, str], ...]) -> QaWriteResult:
        self.items = items
        return QaWriteResult(question_ids=("a" * 32,), created_count=1, updated_count=0)

    def list_question_answers(self, *, limit: int, offset: int) -> list[object]:
        return []

    def delete_many(self, question_ids: list[str]) -> int:
        return len(question_ids)


class FakeCache:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def warmup(self, _: FakeMysql) -> int:
        if self.fail:
            raise RuntimeError("redis unavailable")
        return 1


def testDeduplicatesBatchByQuestionKeepingLastAnswer() -> None:
    mysql = FakeMysql()
    service = QaAdminService(mysql, FakeCache())  # type: ignore[arg-type]

    service.import_items((("问题", "旧答案"), ("问题", "新答案")))

    assert mysql.items == (("问题", "新答案"),)


def testRaisesWhenMySqlCommittedButRedisRefreshFails() -> None:
    service = QaAdminService(FakeMysql(), FakeCache(fail=True))  # type: ignore[arg-type]

    with pytest.raises(CacheSynchronizationError, match="Redis"):
        service.upsert("问题", "答案")
