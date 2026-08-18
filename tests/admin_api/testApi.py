from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from admin_api.app import AdminApiServices, create_app
from admin_api.job_repository import IngestionJob
from mysql_qa.models import QaWriteResult, QuestionAnswer


class FakeMysqlClient:
    def initialize_schema(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeQaService:
    def upsert(self, question: str, answer: str) -> tuple[QaWriteResult, int]:
        assert (question, answer) == ("退货流程", "订单页申请")
        return QaWriteResult(("a" * 32,), 1, 0), 1

    def import_items(self, _: list[tuple[str, str]]) -> tuple[QaWriteResult, int]:
        return QaWriteResult(("a" * 32,), 1, 0), 1

    def list_items(self, *, limit: int, offset: int) -> list[QuestionAnswer]:
        assert (limit, offset) == (50, 0)
        return [QuestionAnswer("a" * 32, "退货流程", "订单页申请")]

    def delete(self, question_ids: list[str]) -> tuple[int, int]:
        return len(question_ids), 0


class FakeJobs:
    def __init__(self, source: Path) -> None:
        self.source = source
        now = datetime.now(timezone.utc)
        self.job = IngestionJob("b" * 32, "manual.txt", str(source), "PENDING", None, 0, None, now, now)

    def initialize_schema(self) -> None:
        return None

    def mark_interrupted_jobs_failed(self) -> int:
        return 0

    def close(self) -> None:
        return None

    def create(self, _: str, filename: str, source_path: str) -> IngestionJob:
        self.job = IngestionJob(
            "b" * 32,
            filename,
            source_path,
            "PENDING",
            None,
            0,
            None,
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
        )
        return self.job


class FakeDocuments:
    def __init__(self, jobs: FakeJobs) -> None:
        self.jobs = jobs
        self.enqueued: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)

    def get(self, _: str) -> IngestionJob:
        return self.jobs.job

    def list(self, **_: int) -> list[IngestionJob]:
        return [self.jobs.job]

    def delete(self, _: str) -> IngestionJob:
        return self.jobs.job

    def shutdown(self) -> None:
        return None


def testLocalAdminApiServesQaAndQueuesDocument(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    jobs = FakeJobs(uploads / "unused.txt")
    documents = FakeDocuments(jobs)
    app = create_app(
        services=AdminApiServices(
            mysql_client=FakeMysqlClient(),  # type: ignore[arg-type]
            qa_service=FakeQaService(),  # type: ignore[arg-type]
            jobs=jobs,  # type: ignore[arg-type]
            documents=documents,  # type: ignore[arg-type]
        ),
        uploads_dir=uploads,
    )

    with TestClient(app) as client:
        qa_response = client.post("/admin/qa", json={"question": "退货流程", "answer": "订单页申请"})
        listed_response = client.get("/admin/qa")
        document_response = client.post(
            "/admin/documents",
            files={"file": ("manual.txt", b"knowledge", "text/plain")},
        )

    assert qa_response.status_code == 201
    assert listed_response.json()["items"][0]["answer"] == "订单页申请"
    assert document_response.status_code == 202
    assert documents.enqueued == ["b" * 32]
