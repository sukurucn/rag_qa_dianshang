"""Redis 问题缓存连接与 MySQL warmup。"""

from __future__ import annotations

from typing import Any

import redis

from base.config import Settings
from base.logger import get_logger
from mysql_qa.models import QuestionAnswer
from mysql_qa.mysql_client import MysqlQaClient


class RedisQuestionCache:
    """只缓存问题 ID 到问题文本的 Redis Hash。"""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self._logger = get_logger("mysql_qa.redis_client")

    def connect(self) -> None:
        """创建并检查 Redis 连接。"""
        if self._client is not None:
            return
        self._logger.info(
            "Connecting to Redis for mysql_qa: host=%s port=%s db=%s",
            self._settings.redis_host,
            self._settings.redis_port,
            self._settings.redis_db,
        )
        try:
            self._client = redis.Redis(
                host=self._settings.redis_host,
                port=self._settings.redis_port,
                db=self._settings.redis_db,
                password=self._settings.redis_password.get_secret_value(),
                decode_responses=True,
            )
            self._client.ping()
            self._logger.info("Redis connection for mysql_qa established")
        except Exception:
            self._logger.exception("Unable to connect to Redis for mysql_qa")
            self._client = None
            raise

    def warmup(self, mysql_client: MysqlQaClient) -> int:
        """将 MySQL 全部问题原子替换到 Redis，返回加载数量。"""
        client = self._require_client()
        self._logger.info("Starting mysql_qa Redis warmup")
        try:
            questions = mysql_client.list_questions()
            mapping = {item.question_id: item.question for item in questions}
            with client.pipeline(transaction=True) as pipeline:
                pipeline.delete(self._settings.mysql_qa_redis_key)
                if mapping:
                    pipeline.hset(self._settings.mysql_qa_redis_key, mapping=mapping)
                pipeline.execute()
            self._logger.info("Completed mysql_qa Redis warmup: count=%s", len(mapping))
            return len(mapping)
        except Exception:
            self._logger.exception("Failed mysql_qa Redis warmup")
            raise

    def get_questions(self) -> list[QuestionAnswer]:
        """返回当前 Redis 缓存中的所有问题。"""
        client = self._require_client()
        try:
            values: dict[str, str] = client.hgetall(self._settings.mysql_qa_redis_key)
            return [QuestionAnswer(question_id=key, question=value) for key, value in values.items()]
        except Exception:
            self._logger.exception("Failed to read mysql_qa questions from Redis")
            raise

    def _require_client(self) -> Any:
        """返回可用 Redis 客户端，供内部缓存操作复用。"""
        self.connect()
        assert self._client is not None
        return self._client
