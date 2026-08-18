from __future__ import annotations

from typing_extensions import Self

from mysql_qa.models import QuestionAnswer
from mysql_qa.redis_client import RedisQuestionCache
from tests.mysql_qa.testMysqlClient import make_settings


class FakePipeline:
    def __init__(self, redis_client: FakeRedis) -> None:
        self._redis_client = redis_client
        self._mapping: dict[str, str] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def delete(self, key: str) -> None:
        return None

    def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def execute(self) -> None:
        self._redis_client.values = self._mapping


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def pipeline(self, transaction: bool) -> FakePipeline:
        assert transaction is True
        return FakePipeline(self)

    def hgetall(self, key: str) -> dict[str, str]:
        return self.values


class FakeMysqlClient:
    def list_questions(self) -> list[QuestionAnswer]:
        return [
            QuestionAnswer(question_id="id-1", question="如何退款？"),
            QuestionAnswer(question_id="id-2", question="多久发货？"),
        ]


def testWarmsAllMysqlQuestionsIntoRedis() -> None:
    redis_client = FakeRedis()
    cache = RedisQuestionCache(make_settings(), client=redis_client)

    assert cache.warmup(FakeMysqlClient()) == 2  # type: ignore[arg-type]
    assert [item.question_id for item in cache.get_questions()] == ["id-1", "id-2"]
