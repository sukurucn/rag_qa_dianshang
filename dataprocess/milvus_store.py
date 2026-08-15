"""文档块写入 Milvus 的显式 schema 适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pymilvus import DataType, MilvusClient

from base.config import Settings
from dataprocess.models import Chunk, Embedding

COLLECTION_NAME = "document_chunks_v2"
DENSE_VECTOR_DIMENSION = 1024
DENSE_INDEX_NAME = "dense_vector_ivf"
SPARSE_INDEX_NAME = "sparse_vector_inverted"


class ChunkStore(Protocol):
    """可替换的文档块存储接口。"""

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """持久化已向量化的块。"""


class DocumentChunkStore(ChunkStore, Protocol):
    """支持按文档清理的块存储接口。"""

    def delete_document(self, document_id: str) -> None:
        """删除一个 document_id 对应的全部父子块。"""


class MilvusDocumentStore:
    """将父子块连同 dense/sparse 向量写入单个 Milvus collection。"""

    def __init__(
        self,
        settings: Settings,
        collection_name: str = COLLECTION_NAME,
        client: Any | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._settings = settings
        self._client: Any | None = client

    def _require_client(self) -> Any:
        """在首次实际读写时连接 Milvus，避免 API 模块导入时阻塞。"""
        if self._client is None:
            self._client = MilvusClient(
                uri=self._settings.milvus_uri,
                token=self._settings.milvus_token,
                db_name=self._settings.milvus_database,
            )
        return self._client

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        """确保 collection 存在后幂等 upsert 所有块。"""
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        if not chunks:
            return
        self._ensure_collection()
        entities = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "parent_id": chunk.parent_id,
                "source": chunk.source,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type,
                "title_path": chunk.title_path,
                "block_type": chunk.block_type,
                "parent_ids": list(chunk.parent_ids),
                "dense_vector": embedding.dense_vector,
                "sparse_vector": embedding.sparse_vector,
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._require_client().upsert(collection_name=self._collection_name, data=entities)

    def delete_document(self, document_id: str) -> None:
        """按现有 document_id 字段删除全部父块和子块，不改动 schema。"""
        if len(document_id) != 32 or any(character not in "0123456789abcdef" for character in document_id):
            raise ValueError("document_id must be a lowercase MD5 hash")
        client = self._require_client()
        if not client.has_collection(collection_name=self._collection_name):
            return
        client.delete(
            collection_name=self._collection_name,
            filter=f'document_id == "{document_id}"',
        )

    def _ensure_collection(self) -> None:
        client = self._require_client()
        if client.has_collection(collection_name=self._collection_name):
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=32)
        schema.add_field("document_id", DataType.VARCHAR, max_length=32)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=32)
        schema.add_field("source", DataType.VARCHAR, max_length=2048)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
        schema.add_field("title_path", DataType.VARCHAR, max_length=4096)
        schema.add_field("block_type", DataType.VARCHAR, max_length=16)
        schema.add_field("parent_ids", DataType.JSON)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DENSE_VECTOR_DIMENSION)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        indexes = client.prepare_index_params()
        indexes.add_index(
            field_name="dense_vector",
            index_name=DENSE_INDEX_NAME,
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": self._settings.milvus_dense_index_nlist},
        )
        indexes.add_index(
            field_name="sparse_vector",
            index_name=SPARSE_INDEX_NAME,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=indexes,
        )
