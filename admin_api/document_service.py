"""单线程异步文档入库和可审计删除流程。"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

from admin_api.job_repository import IngestionJob, IngestionJobRepository
from base.config import Settings
from base.logger import get_logger
from dataprocess.data_processor import data_processor
from dataprocess.milvus_store import DocumentChunkStore
from dataprocess.models import ProcessingReport

Processor = Callable[..., ProcessingReport]


class DocumentDeletionError(RuntimeError):
    """无法安全删除文档任务时抛出。"""


class DocumentIngestionService:
    """将耗时且占用 GPU 的处理串行化，防止并发向量化导致 OOM。"""

    def __init__(
        self,
        settings: Settings,
        jobs: IngestionJobRepository,
        store: DocumentChunkStore,
        *,
        processor: Processor = data_processor,
        uploads_dir: Path,
        executor: Executor | None = None,
    ) -> None:
        self._settings = settings
        self._jobs = jobs
        self._store = store
        self._processor = processor
        self._uploads_dir = uploads_dir.resolve()
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="document-ingestion")
        self._owns_executor = executor is None
        self._logger = get_logger("admin_api.document_service")

    def enqueue(self, job_id: str) -> None:
        """将已持久化任务交给唯一后台 worker；服务重启时不会重新投递。"""
        self._executor.submit(self._run_job, job_id)

    def get(self, job_id: str) -> IngestionJob | None:
        return self._jobs.get(job_id)

    def list(self, *, limit: int, offset: int) -> list[IngestionJob]:
        return self._jobs.list(limit=limit, offset=offset)

    def delete(self, job_id: str) -> IngestionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "SUCCEEDED":
            raise DocumentDeletionError("仅允许删除已完成的文档任务")
        if job.document_id is None:
            raise DocumentDeletionError("已完成任务缺少 document_id，无法安全删除")
        source_path = Path(job.source_path).resolve()
        if self._uploads_dir not in source_path.parents:
            raise DocumentDeletionError("任务原始文件不在 uploads 目录，拒绝删除")
        self._store.delete_document(job.document_id)
        try:
            source_path.unlink(missing_ok=True)
        except OSError as error:
            self._logger.exception("Failed to remove uploaded file: %s", source_path)
            raise DocumentDeletionError("Milvus 已清理，但无法删除本地上传文件") from error
        if not self._jobs.mark_deleted(job_id):
            raise DocumentDeletionError("任务状态已改变，无法标记删除")
        deleted = self._jobs.get(job_id)
        assert deleted is not None
        return deleted

    def shutdown(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=False)

    def _run_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or not self._jobs.mark_running(job_id):
            return
        try:
            report = self._processor(job.source_path, app_settings=self._settings)
            if report.processed_documents != 1 or len(report.documents) != 1:
                reason = "; ".join(report.skipped_files) or "文档未产生可入库内容"
                raise RuntimeError(reason)
            document = report.documents[0]
            if not self._jobs.mark_succeeded(job_id, document.document_id, document.stored_chunks):
                self._logger.warning("Document job did not transition to SUCCEEDED: %s", job_id)
        except Exception as error:
            self._logger.exception("Document ingestion job failed: %s", job_id)
            self._jobs.mark_failed(job_id, str(error))
