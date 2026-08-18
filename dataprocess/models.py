"""文档处理过程使用的领域模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Embedding:
    """同一文本的稠密和稀疏表示。"""

    dense_vector: list[float]
    sparse_vector: dict[int, float]


@dataclass(frozen=True)
class Chunk:
    """将要写入 Milvus 的结构化父块或子块。"""

    chunk_id: str
    document_id: str
    parent_id: str
    source: str
    text: str
    chunk_type: str
    title_path: str
    block_type: str
    parent_ids: tuple[str, ...]

    @property
    def embedding_text(self) -> str:
        """返回包含文档和章节上下文、专供向量化的文本。"""
        context = f"文档: {self.source}\n章节: {self.title_path}"
        return f"{context}\n内容: {self.text}"


@dataclass(frozen=True)
class ProcessingReport:
    """一次文档处理调用的汇总结果。"""

    processed_documents: int
    stored_chunks: int
    skipped_files: tuple[str, ...]
    documents: tuple["ProcessedDocument", ...] = ()


@dataclass(frozen=True)
class ProcessedDocument:
    """已成功写入 Milvus 的单个文档，供任务审计与删除操作追踪。"""

    source: str
    document_id: str
    stored_chunks: int
