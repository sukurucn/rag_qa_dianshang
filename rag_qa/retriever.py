"""Milvus 稠密和稀疏子块检索及父块回查。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pymilvus import MilvusClient  # type: ignore[import-untyped]

from base.config import Settings
from base.logger import get_logger
from dataprocess.embedding import BgeM3Embedder
from dataprocess.milvus_store import COLLECTION_NAME
from rag_qa.models import Embedder, ParentChunk, RetrievalReport

CHILD_OUTPUT_FIELDS = ["chunk_id", "parent_id"]
PARENT_OUTPUT_FIELDS = ["chunk_id", "chunk_type", "source", "title_path", "text"]
CHILD_FILTER = 'chunk_type == "child"'


class MilvusHybridRetriever:
    """分别获取 BGE-M3 dense/sparse TopK 子块，再批量读取父块。"""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        client: Any | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._settings = settings
        self._embedder = embedder or BgeM3Embedder()
        self._client = client or MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            db_name=settings.milvus_database,
        )
        self._collection_name = collection_name
        self._logger = get_logger("rag_qa.retriever")

    def retrieve(self, queries: Sequence[str]) -> RetrievalReport:
        """按 dense/sparse 各 TopK 查子块，随后按 parent_id 去重回查父块。"""
        normalized_queries = tuple(query.strip() for query in queries if query.strip())
        if not normalized_queries:
            return RetrievalReport(parents=(), dense_hit_count=0, sparse_hit_count=0)
        embeddings = self._embedder.embed(normalized_queries)
        if len(embeddings) != len(normalized_queries):
            raise RuntimeError("BGE-M3 embedding count does not match query count")

        dense_hits: list[dict[str, Any]] = []
        sparse_hits: list[dict[str, Any]] = []
        for embedding in embeddings:
            dense_hits.extend(
                self._search(
                    embedding.dense_vector,
                    "dense_vector",
                    "COSINE",
                    {"nprobe": self._settings.milvus_dense_search_nprobe},
                )
            )
            sparse_hits.extend(self._search(embedding.sparse_vector, "sparse_vector", "IP", {}))
        parent_ids = tuple(
            dict.fromkeys(
                str(hit["parent_id"])
                for hit in [*dense_hits, *sparse_hits]
                if hit.get("parent_id")
            )
        )
        parents = self._load_parents(parent_ids)
        self._logger.info(
            "rag retrieval: queries=%s dense_hits=%s sparse_hits=%s parents=%s",
            len(normalized_queries),
            len(dense_hits),
            len(sparse_hits),
            len(parents),
        )
        return RetrievalReport(
            parents=parents,
            dense_hit_count=len(dense_hits),
            sparse_hit_count=len(sparse_hits),
        )

    def _search(
        self,
        vector: list[float] | dict[int, float],
        field: str,
        metric: str,
        params: dict[str, int],
    ) -> list[dict[str, Any]]:
        result = self._client.search(
            collection_name=self._collection_name,
            data=[vector],
            filter=CHILD_FILTER,
            limit=self._settings.rag_top_k,
            output_fields=CHILD_OUTPUT_FIELDS,
            anns_field=field,
            search_params={"metric_type": metric, "params": params},
        )
        return self._flatten_search_result(result)

    def _load_parents(self, parent_ids: Sequence[str]) -> tuple[ParentChunk, ...]:
        if not parent_ids:
            return ()
        rows: Sequence[dict[str, Any]] = self._client.get(
            collection_name=self._collection_name,
            ids=list(parent_ids),
            output_fields=PARENT_OUTPUT_FIELDS,
        )
        parents: list[ParentChunk] = []
        seen_ids: set[str] = set()
        for row in rows:
            chunk_id = str(row.get("chunk_id", ""))
            if row.get("chunk_type") != "parent" or not chunk_id or chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            parents.append(
                ParentChunk(
                    chunk_id=chunk_id,
                    source=str(row.get("source", "")),
                    title_path=str(row.get("title_path", "")),
                    text=str(row.get("text", "")),
                )
            )
        return tuple(parents)

    @staticmethod
    def _flatten_search_result(result: Sequence[Any]) -> list[dict[str, Any]]:
        """兼容 MilvusClient 单查询的嵌套和非嵌套返回形式。"""
        if not result:
            return []
        first_item = result[0]
        if isinstance(first_item, list):
            return [item for group in result for item in group if isinstance(item, dict)]
        return [item for item in result if isinstance(item, dict)]
