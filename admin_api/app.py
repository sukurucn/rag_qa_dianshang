"""本机知识库管理 FastAPI 应用。"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status

from admin_api.document_service import DocumentDeletionError, DocumentIngestionService
from admin_api.import_parser import QaImportValidationError, parse_qa_import
from admin_api.job_repository import IngestionJob, IngestionJobRepository
from admin_api.qa_service import CacheSynchronizationError, QaAdminService
from admin_api.schemas import (
    IngestionJobResponse,
    QaDeleteRequest,
    QaItem,
    QaListResponse,
    QaPayload,
    QaWriteResponse,
)
from base.config import PROJECT_ROOT, Settings, settings
from base.logger import get_logger
from dataprocess.file_converter import SUPPORTED_SUFFIXES
from dataprocess.milvus_store import MilvusDocumentStore
from mysql_qa.mysql_client import MysqlQaClient
from mysql_qa.redis_client import RedisQuestionCache

MAX_DOCUMENT_BYTES = 100 * 1024 * 1024
UPLOADS_DIR = PROJECT_ROOT / "uploads"
_LOGGER = get_logger("admin_api")


@dataclass
class AdminApiServices:
    """应用依赖容器；测试可替换为无网络的实现。"""

    mysql_client: MysqlQaClient
    qa_service: QaAdminService
    jobs: IngestionJobRepository
    documents: DocumentIngestionService

    def start(self) -> None:
        self.mysql_client.initialize_schema()
        self.jobs.initialize_schema()
        interrupted = self.jobs.mark_interrupted_jobs_failed()
        if interrupted:
            _LOGGER.warning("Marked interrupted document jobs as failed: count=%s", interrupted)

    def stop(self) -> None:
        self.documents.shutdown()
        self.jobs.close()
        self.mysql_client.close()


def create_default_services(app_settings: Settings = settings) -> AdminApiServices:
    """构建运行时依赖，所有网络连接均延迟至应用启动后。"""
    mysql_client = MysqlQaClient(app_settings)
    cache = RedisQuestionCache(app_settings)
    jobs = IngestionJobRepository(app_settings)
    store = MilvusDocumentStore(app_settings)
    documents = DocumentIngestionService(
        app_settings,
        jobs,
        store,
        uploads_dir=UPLOADS_DIR,
    )
    return AdminApiServices(
        mysql_client=mysql_client,
        qa_service=QaAdminService(mysql_client, cache),
        jobs=jobs,
        documents=documents,
    )


def create_app(
    *,
    app_settings: Settings = settings,
    services: AdminApiServices | None = None,
    uploads_dir: Path = UPLOADS_DIR,
) -> FastAPI:
    """创建仅适合本机绑定的管理 API；认证应由外层网关处理。"""
    active_services = services or create_default_services(app_settings)
    active_uploads_dir = uploads_dir.resolve()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        active_uploads_dir.mkdir(parents=True, exist_ok=True)
        active_services.start()
        try:
            yield
        finally:
            active_services.stop()

    app = FastAPI(title="RAG 本地知识库管理 API", version="0.1.0", lifespan=lifespan)
    app.state.services = active_services

    @app.post("/admin/qa", response_model=QaWriteResponse, status_code=status.HTTP_201_CREATED)
    def upsert_qa(payload: QaPayload) -> QaWriteResponse:
        try:
            result, warmed_count = active_services.qa_service.upsert(payload.question, payload.answer)
        except CacheSynchronizationError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        return QaWriteResponse(
            question_ids=list(result.question_ids),
            created_count=result.created_count,
            updated_count=result.updated_count,
            deleted_count=0,
            redis_warmed_count=warmed_count,
        )

    @app.post("/admin/qa/import", response_model=QaWriteResponse)
    async def import_qa(file: UploadFile = File(...)) -> QaWriteResponse:  # noqa: B008
        filename = Path(file.filename or "").name
        try:
            items = parse_qa_import(filename, await file.read())
            result, warmed_count = active_services.qa_service.import_items(items)
        except QaImportValidationError as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
        except CacheSynchronizationError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        return QaWriteResponse(
            question_ids=list(result.question_ids),
            created_count=result.created_count,
            updated_count=result.updated_count,
            deleted_count=0,
            redis_warmed_count=warmed_count,
        )

    @app.get("/admin/qa", response_model=QaListResponse)
    def list_qa(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> QaListResponse:
        records = active_services.qa_service.list_items(limit=limit, offset=offset)
        return QaListResponse(
            items=[
                QaItem(question_id=item.question_id, question=item.question, answer=item.answer or "")
                for item in records
            ],
            limit=limit,
            offset=offset,
        )

    @app.delete("/admin/qa", response_model=QaWriteResponse)
    def delete_qa(payload: QaDeleteRequest) -> QaWriteResponse:
        try:
            deleted_count, warmed_count = active_services.qa_service.delete(payload.question_ids)
        except CacheSynchronizationError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        return QaWriteResponse(
            question_ids=payload.question_ids,
            created_count=0,
            updated_count=0,
            deleted_count=deleted_count,
            redis_warmed_count=warmed_count,
        )

    @app.post("/admin/documents", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
    async def upload_document(file: UploadFile = File(...)) -> IngestionJobResponse:  # noqa: B008
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if not filename or suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="仅支持 TXT、MD、DOCX、PDF、PPT、PPTX 文件",
            )
        active_uploads_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        stored_path = active_uploads_dir / f"{job_id}_{filename}"
        try:
            written = await _save_upload(file, stored_path)
            if written == 0:
                raise ValueError("上传文件不能为空")
            job = active_services.jobs.create(job_id, filename, str(stored_path))
        except ValueError as error:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(error)) from error
        except Exception:
            stored_path.unlink(missing_ok=True)
            _LOGGER.exception("Failed to persist uploaded document metadata")
            raise
        active_services.documents.enqueue(job.job_id)
        return _job_response(job)

    @app.get("/admin/documents/jobs/{job_id}", response_model=IngestionJobResponse)
    def get_document_job(job_id: str) -> IngestionJobResponse:
        job = active_services.documents.get(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        return _job_response(job)

    @app.get("/admin/documents", response_model=list[IngestionJobResponse])
    def list_document_jobs(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[IngestionJobResponse]:
        return [_job_response(job) for job in active_services.documents.list(limit=limit, offset=offset)]

    @app.delete("/admin/documents/{job_id}", response_model=IngestionJobResponse)
    def delete_document(job_id: str) -> IngestionJobResponse:
        try:
            return _job_response(active_services.documents.delete(job_id))
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在") from error
        except DocumentDeletionError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    return app


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_DOCUMENT_BYTES:
                    raise ValueError("上传文件不能超过 100MB")
                output.write(chunk)
    finally:
        await upload.close()
    return written


def _job_response(job: IngestionJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        job_id=job.job_id,
        original_filename=job.original_filename,
        status=job.status,
        document_id=job.document_id,
        stored_chunks=job.stored_chunks,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
