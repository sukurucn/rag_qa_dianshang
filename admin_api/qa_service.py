"""MySQL QA 写入与 Redis 全量热加载的事务边界。"""

from __future__ import annotations

from collections.abc import Iterable

from base.logger import get_logger
from mysql_qa.models import QaWriteResult, QuestionAnswer
from mysql_qa.mysql_client import MysqlQaClient
from mysql_qa.redis_client import RedisQuestionCache


class CacheSynchronizationError(RuntimeError):
    """MySQL 已提交，但 Redis 全量同步失败。"""


class QaAdminService:
    """将 QA 的持久化与缓存刷新作为一条管理工作流执行。"""

    def __init__(self, mysql_client: MysqlQaClient, redis_cache: RedisQuestionCache) -> None:
        self._mysql_client = mysql_client
        self._redis_cache = redis_cache
        self._logger = get_logger("admin_api.qa_service")

    def upsert(self, question: str, answer: str) -> tuple[QaWriteResult, int]:
        return self.import_items(((question, answer),))

    def import_items(self, items: Iterable[tuple[str, str]]) -> tuple[QaWriteResult, int]:
        """先一次性提交 MySQL，再原子替换 Redis Hash。"""
        latest_by_question: dict[str, str] = {}
        for question, answer in items:
            clean_question, clean_answer = question.strip(), answer.strip()
            if not clean_question or not clean_answer:
                raise ValueError("question and answer must not be empty")
            latest_by_question[clean_question] = clean_answer
        result = self._mysql_client.bulk_upsert(tuple(latest_by_question.items()))
        try:
            warmed_count = self._redis_cache.warmup(self._mysql_client)
        except Exception as error:
            self._logger.exception("MySQL committed but Redis warmup failed")
            raise CacheSynchronizationError("MySQL 已提交，但 Redis 缓存刷新失败") from error
        return result, warmed_count

    def list_items(self, *, limit: int, offset: int) -> list[QuestionAnswer]:
        return self._mysql_client.list_question_answers(limit=limit, offset=offset)

    def delete(self, question_ids: list[str]) -> tuple[int, int]:
        deleted_count = self._mysql_client.delete_many(question_ids)
        try:
            warmed_count = self._redis_cache.warmup(self._mysql_client)
        except Exception as error:
            self._logger.exception("MySQL deletion committed but Redis warmup failed")
            raise CacheSynchronizationError("MySQL 已提交，但 Redis 缓存刷新失败") from error
        return deleted_count, warmed_count
