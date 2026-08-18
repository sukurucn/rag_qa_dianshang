"""持久化管理问答链路的可开关模块。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Protocol

import pymysql  # type: ignore[import-untyped]

from base.config import Settings
from base.logger import get_logger


@dataclass(frozen=True)
class FeatureFlags:
    """用户问答时生效的模块状态，默认均开启。"""

    faq_enabled: bool = True
    classifier_enabled: bool = True


class FeatureFlagStore(Protocol):
    """功能开关的持久化边界。"""

    def initialize(self) -> None: ...

    def load(self) -> FeatureFlags: ...

    def save(self, flags: FeatureFlags) -> None: ...

    def close(self) -> None: ...


class MysqlFeatureFlagStore:
    """将开关状态独立保存到 MySQL，避免服务重启后丢失。"""

    _TABLE_NAME = "query_feature_flags"

    def __init__(self, settings: Settings, connection_factory: Callable[..., Any] = pymysql.connect) -> None:
        self._settings = settings
        self._connection_factory = connection_factory
        self._connection: Any | None = None
        self._logger = get_logger("query_api.feature_flags")

    def initialize(self) -> None:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS `{self._TABLE_NAME}` (
                        feature_name VARCHAR(64) NOT NULL,
                        enabled BOOLEAN NOT NULL,
                        updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                            ON UPDATE CURRENT_TIMESTAMP(6),
                        PRIMARY KEY (feature_name)
                    ) CHARACTER SET {self._settings.mysql_charset}
                    """
                )
                cursor.executemany(
                    f"INSERT IGNORE INTO `{self._TABLE_NAME}` (feature_name, enabled) VALUES (%s, %s)",
                    (("faq_enabled", True), ("classifier_enabled", True)),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to initialize feature-flag table")
            raise

    def load(self) -> FeatureFlags:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT feature_name, enabled FROM `{self._TABLE_NAME}` "
                    "WHERE feature_name IN (%s, %s)",
                    ("faq_enabled", "classifier_enabled"),
                )
                rows = {str(name): bool(enabled) for name, enabled in cursor.fetchall()}
            connection.rollback()
            return FeatureFlags(
                faq_enabled=rows.get("faq_enabled", True),
                classifier_enabled=rows.get("classifier_enabled", True),
            )
        except Exception:
            self._logger.exception("Failed to load feature flags")
            raise

    def save(self, flags: FeatureFlags) -> None:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO `{self._TABLE_NAME}` (feature_name, enabled)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE enabled = VALUES(enabled)
                    """,
                    (("faq_enabled", flags.faq_enabled), ("classifier_enabled", flags.classifier_enabled)),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to persist feature flags")
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


class FeatureFlagService:
    """缓存已持久化的开关，并为请求提供一致快照。"""

    def __init__(self, store: FeatureFlagStore) -> None:
        self._store = store
        self._flags = FeatureFlags()
        self._lock = Lock()
        self._logger = get_logger("query_api.feature_flags")

    def start(self) -> None:
        try:
            self._store.initialize()
            with self._lock:
                self._flags = self._store.load()
            self._logger.info("Feature flags loaded: %s", self._flags)
        except Exception:
            self._logger.exception("Feature flags unavailable; using default enabled state")

    def current(self) -> FeatureFlags:
        with self._lock:
            return self._flags

    def update(
        self,
        *,
        faq_enabled: bool | None,
        classifier_enabled: bool | None,
    ) -> FeatureFlags:
        if faq_enabled is None and classifier_enabled is None:
            raise ValueError("at least one feature flag must be supplied")
        with self._lock:
            updated = replace(
                self._flags,
                faq_enabled=self._flags.faq_enabled if faq_enabled is None else faq_enabled,
                classifier_enabled=(
                    self._flags.classifier_enabled if classifier_enabled is None else classifier_enabled
                ),
            )
            self._store.save(updated)
            self._flags = updated
        self._logger.info("Feature flags updated: %s", updated)
        return updated

    def close(self) -> None:
        self._store.close()
