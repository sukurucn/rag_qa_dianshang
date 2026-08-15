"""文档入库任务的 MySQL 审计存储。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import pymysql  # type: ignore[import-untyped]

from base.config import Settings
from base.logger import get_logger

JobStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "DELETED"]
JOB_TABLE_NAME = "ingestion_jobs"


@dataclass(frozen=True)
class IngestionJob:
    job_id: str
    original_filename: str
    source_path: str
    status: JobStatus
    document_id: str | None
    stored_chunks: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class IngestionJobRepository:
    """独立维护 ingestion_jobs，不修改既有 question_answers 表。"""

    def __init__(self, settings: Settings, connection_factory: Callable[..., Any] = pymysql.connect) -> None:
        self._settings = settings
        self._connection_factory = connection_factory
        self._connection: Any | None = None
        self._logger = get_logger("admin_api.job_repository")

    def connect(self) -> None:
        if self._connection is not None:
            return
        self._connection = self._connection_factory(
            host=self._settings.mysql_host,
            port=self._settings.mysql_port,
            user=self._settings.mysql_user,
            password=self._settings.mysql_password.get_secret_value(),
            database=self._settings.mysql_database,
            charset=self._settings.mysql_charset,
            autocommit=False,
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def initialize_schema(self) -> None:
        statement = f"""
            CREATE TABLE IF NOT EXISTS `{JOB_TABLE_NAME}` (
                job_id CHAR(32) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                source_path VARCHAR(2048) NOT NULL,
                status VARCHAR(16) NOT NULL,
                document_id CHAR(32) NULL,
                stored_chunks INT NOT NULL DEFAULT 0,
                error_message TEXT NULL,
                created_at DATETIME(6) NOT NULL,
                updated_at DATETIME(6) NOT NULL,
                PRIMARY KEY (job_id),
                INDEX idx_ingestion_jobs_created_at (created_at)
            ) CHARACTER SET {self._settings.mysql_charset}
        """
        self._execute_write(statement, ())

    def create(self, job_id: str, original_filename: str, source_path: str) -> IngestionJob:
        now = _database_now()
        self._execute_write(
            f"""
            INSERT INTO `{JOB_TABLE_NAME}`
                (job_id, original_filename, source_path, status, document_id, stored_chunks,
                 error_message, created_at, updated_at)
            VALUES (%s, %s, %s, 'PENDING', NULL, 0, NULL, %s, %s)
            """,
            (job_id, original_filename, source_path, now, now),
        )
        return IngestionJob(job_id, original_filename, source_path, "PENDING", None, 0, None, now, now)

    def get(self, job_id: str) -> IngestionJob | None:
        connection = self._require_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT job_id, original_filename, source_path, status, document_id,
                           stored_chunks, error_message, created_at, updated_at
                    FROM `{JOB_TABLE_NAME}` WHERE job_id = %s""",
                (job_id,),
            )
            row: tuple[Any, ...] | None = cursor.fetchone()
        return self._to_job(row) if row is not None else None

    def list(self, *, limit: int, offset: int) -> list[IngestionJob]:
        connection = self._require_connection()
        with connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT job_id, original_filename, source_path, status, document_id,
                           stored_chunks, error_message, created_at, updated_at
                    FROM `{JOB_TABLE_NAME}` ORDER BY created_at DESC LIMIT %s OFFSET %s""",
                (limit, offset),
            )
            rows: Sequence[tuple[Any, ...]] = cursor.fetchall()
        return [self._to_job(row) for row in rows]

    def mark_running(self, job_id: str) -> bool:
        return self._update_status(job_id, "RUNNING", expected_status="PENDING")

    def mark_succeeded(self, job_id: str, document_id: str, stored_chunks: int) -> bool:
        return self._update_status(
            job_id,
            "SUCCEEDED",
            expected_status="RUNNING",
            document_id=document_id,
            stored_chunks=stored_chunks,
            error_message=None,
        )

    def mark_failed(self, job_id: str, error_message: str) -> bool:
        return self._update_status(
            job_id,
            "FAILED",
            expected_status=("PENDING", "RUNNING"),
            error_message=error_message[:10_000],
        )

    def mark_deleted(self, job_id: str) -> bool:
        return self._update_status(job_id, "DELETED", expected_status="SUCCEEDED")

    def mark_interrupted_jobs_failed(self) -> int:
        connection = self._require_connection()
        now = _database_now()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""UPDATE `{JOB_TABLE_NAME}`
                        SET status = 'FAILED', error_message = %s, updated_at = %s
                        WHERE status IN ('PENDING', 'RUNNING')""",
                    ("服务重启前任务未完成，已停止且不会自动重放。", now),
                )
                updated = int(cursor.rowcount)
            connection.commit()
            return updated
        except Exception:
            connection.rollback()
            self._logger.exception("Failed to mark interrupted ingestion jobs")
            raise

    def _update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        expected_status: JobStatus | tuple[JobStatus, ...],
        document_id: str | None = None,
        stored_chunks: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        expected = (expected_status,) if isinstance(expected_status, str) else expected_status
        assignments = ["status = %s", "updated_at = %s"]
        parameters: list[Any] = [status, _database_now()]
        if document_id is not None:
            assignments.append("document_id = %s")
            parameters.append(document_id)
        if stored_chunks is not None:
            assignments.append("stored_chunks = %s")
            parameters.append(stored_chunks)
        assignments.append("error_message = %s")
        parameters.append(error_message)
        placeholders = ", ".join(["%s"] * len(expected))
        parameters.append(job_id)
        parameters.extend(expected)
        statement = (
            f"UPDATE `{JOB_TABLE_NAME}` SET {', '.join(assignments)} "
            f"WHERE job_id = %s AND status IN ({placeholders})"
        )
        return self._execute_write(statement, tuple(parameters)) == 1

    def _execute_write(self, statement: str, params: tuple[Any, ...]) -> int:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(statement, params)
                row_count = int(cursor.rowcount)
            connection.commit()
            return row_count
        except Exception:
            connection.rollback()
            self._logger.exception("Ingestion job database write failed")
            raise

    def _require_connection(self) -> Any:
        self.connect()
        assert self._connection is not None
        return self._connection

    @staticmethod
    def _to_job(row: tuple[Any, ...]) -> IngestionJob:
        return IngestionJob(
            job_id=str(row[0]),
            original_filename=str(row[1]),
            source_path=str(row[2]),
            status=row[3],
            document_id=str(row[4]) if row[4] is not None else None,
            stored_chunks=int(row[5]),
            error_message=str(row[6]) if row[6] is not None else None,
            created_at=row[7],
            updated_at=row[8],
        )


def _database_now() -> datetime:
    """返回 MySQL DATETIME 可接受的 UTC 无时区时间。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)
