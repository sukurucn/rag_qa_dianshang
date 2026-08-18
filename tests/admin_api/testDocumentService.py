from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from admin_api.document_service import DocumentIngestionService
from admin_api.job_repository import IngestionJob
from base.config import Settings
from dataprocess.models import ProcessedDocument, ProcessingReport


class ImmediateExecutor:
    def submit(self, function: object, *args: object) -> None:
        function(*args)  # type: ignore[operator]


class FakeJobs:
    def __init__(self, job: IngestionJob) -> None:
        self.job = job

    def get(self, _: str) -> IngestionJob:
        return self.job

    def list(self, **_: int) -> list[IngestionJob]:
        return [self.job]

    def mark_running(self, _: str) -> bool:
        self.job = self._replace(status="RUNNING")
        return True

    def mark_succeeded(self, _: str, document_id: str, stored_chunks: int) -> bool:
        self.job = self._replace(status="SUCCEEDED", document_id=document_id, stored_chunks=stored_chunks)
        return True

    def mark_failed(self, _: str, error: str) -> bool:
        self.job = self._replace(status="FAILED", error_message=error)
        return True

    def mark_deleted(self, _: str) -> bool:
        self.job = self._replace(status="DELETED")
        return True

    def _replace(self, **changes: object) -> IngestionJob:
        data = self.job.__dict__ | changes
        return IngestionJob(**data)


class FakeStore:
    def __init__(self) -> None:
        self.deleted_document_id: str | None = None

    def delete_document(self, document_id: str) -> None:
        self.deleted_document_id = document_id


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


def testRunsDocumentThenDeletesItsMilvusChunksAndSource(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source = uploads / "file.txt"
    source.write_text("知识库内容", encoding="utf-8")
    now = datetime.now(timezone.utc)
    jobs = FakeJobs(
        IngestionJob("a" * 32, "file.txt", str(source), "PENDING", None, 0, None, now, now)
    )
    store = FakeStore()

    def processor(_: str, **__: object) -> ProcessingReport:
        return ProcessingReport(
            processed_documents=1,
            stored_chunks=6,
            skipped_files=(),
            documents=(ProcessedDocument(str(source), "b" * 32, 6),),
        )

    service = DocumentIngestionService(
        make_settings(),
        jobs,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        processor=processor,
        uploads_dir=uploads,
        executor=ImmediateExecutor(),  # type: ignore[arg-type]
    )

    service.enqueue("a" * 32)
    deleted = service.delete("a" * 32)

    assert deleted.status == "DELETED"
    assert store.deleted_document_id == "b" * 32
    assert not source.exists()
