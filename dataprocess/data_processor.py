"""面向本地文件的文档转换、切块、向量化和入库入口。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from base.config import Settings, settings
from base.logger import get_logger
from dataprocess.chunker import create_chunks, normalize_text
from dataprocess.embedding import BgeM3Embedder, EmbeddingProvider
from dataprocess.file_converter import (
    LOCAL_SUFFIXES,
    MINERU_SUFFIXES,
    SUPPORTED_SUFFIXES,
    MinerUParser,
    convert_local_document,
)
from dataprocess.infrastructure import ensure_infrastructure
from dataprocess.milvus_store import ChunkStore, MilvusDocumentStore
from dataprocess.mineru_client import MinerUClient
from dataprocess.models import ProcessedDocument, ProcessingReport
from dataprocess.pdf_splitter import split_pdf

logger = get_logger("dataprocess")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def data_processor(
    input_path: str | Path,
    *,
    start_infrastructure: bool = True,
    app_settings: Settings = settings,
    mineru_parser: MinerUParser | None = None,
    embedder: EmbeddingProvider | None = None,
    store: ChunkStore | None = None,
) -> ProcessingReport:
    """处理单文件或目录，并把父子块及双向量幂等写入 Milvus。"""
    path = Path(input_path).resolve()
    files = _discover_files(path)
    if not files:
        raise ValueError(f"No supported documents were found at: {path}")
    if (
        mineru_parser is None
        and any(candidate.suffix.lower() in MINERU_SUFFIXES for candidate in files)
        and app_settings.mineru_api_key is None
    ):
        raise ValueError("MINERU_API_KEY is required before processing PDF or PPT documents.")
    if start_infrastructure:
        ensure_infrastructure(PROJECT_ROOT)

    active_embedder = embedder or BgeM3Embedder()
    active_store = store or MilvusDocumentStore(app_settings)
    active_mineru = mineru_parser
    processed_documents = 0
    stored_chunks = 0
    skipped_files: list[str] = []
    processed: list[ProcessedDocument] = []

    for document_path in files:
        try:
            markdown = _convert_document(document_path, app_settings, active_mineru)
            document_chunks = create_chunks(str(document_path), markdown)
            if not document_chunks:
                skipped_files.append(str(document_path))
                logger.warning("Skipping empty document: %s", document_path)
                continue
            embeddings = active_embedder.embed([chunk.embedding_text for chunk in document_chunks])
            active_store.upsert(document_chunks, embeddings)
            processed_documents += 1
            stored_chunks += len(document_chunks)
            processed.append(
                ProcessedDocument(
                    source=str(document_path),
                    document_id=document_chunks[0].document_id,
                    stored_chunks=len(document_chunks),
                )
            )
            logger.info("Stored %s chunks from %s", len(document_chunks), document_path)
        except (OSError, RuntimeError, ValueError) as error:
            logger.exception("Failed to process document: %s", document_path)
            skipped_files.append(f"{document_path}: {error}")
    return ProcessingReport(
        processed_documents=processed_documents,
        stored_chunks=stored_chunks,
        skipped_files=tuple(skipped_files),
        documents=tuple(processed),
    )


def _discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _convert_document(path: Path, app_settings: Settings, mineru_parser: MinerUParser | None) -> str:
    suffix = path.suffix.lower()
    if suffix in LOCAL_SUFFIXES:
        return normalize_text(convert_local_document(path))
    if suffix == ".pdf":
        with TemporaryDirectory(prefix="rag-agentic-pdf-") as directory:
            parts = split_pdf(path, Path(directory))
            parser = mineru_parser or _create_mineru_parser(app_settings)
            return normalize_text("\n\n".join(parser.parse_files([part.path for part in parts])))
    if suffix in MINERU_SUFFIXES:
        parser = mineru_parser or _create_mineru_parser(app_settings)
        return normalize_text("\n\n".join(parser.parse_files([path])))
    raise ValueError(f"Unsupported document suffix: {suffix}")


def _create_mineru_parser(app_settings: Settings) -> MinerUClient:
    if app_settings.mineru_api_key is None:
        raise ValueError("MINERU_API_KEY is required before processing PDF or PPT documents.")
    return MinerUClient(app_settings.mineru_api_key.get_secret_value())
