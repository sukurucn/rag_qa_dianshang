"""显式重建 Milvus 稠密 IVF_FLAT 索引的维护命令。"""

from __future__ import annotations

import argparse
from typing import Any

from pymilvus import MilvusClient  # type: ignore[import-untyped]

from base.config import Settings, settings
from base.logger import get_logger
from dataprocess.milvus_store import COLLECTION_NAME, DENSE_INDEX_NAME


class MilvusDenseIndexManager:
    """在维护窗口显式将稠密向量索引重建为 IVF_FLAT。"""

    def __init__(
        self,
        app_settings: Settings,
        *,
        client: Any | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._settings = app_settings
        self._collection_name = collection_name
        self._client = client or MilvusClient(
            uri=app_settings.milvus_uri,
            token=app_settings.milvus_token,
            db_name=app_settings.milvus_database,
        )
        self._logger = get_logger("dataprocess.milvus_index")

    def rebuild_dense_index(self) -> None:
        """释放 collection，替换稠密索引，随后重新加载 collection。"""
        if not self._client.has_collection(collection_name=self._collection_name):
            raise RuntimeError(f"Milvus collection does not exist: {self._collection_name}")

        index_names = list(
            self._client.list_indexes(
                collection_name=self._collection_name,
                field_name="dense_vector",
            )
        )
        self._logger.info(
            "Rebuilding Milvus dense index: collection=%s old_indexes=%s nlist=%s",
            self._collection_name,
            index_names,
            self._settings.milvus_dense_index_nlist,
        )
        self._client.release_collection(collection_name=self._collection_name)
        for index_name in index_names:
            self._client.drop_index(
                collection_name=self._collection_name,
                index_name=index_name,
            )

        indexes = self._client.prepare_index_params()
        indexes.add_index(
            field_name="dense_vector",
            index_name=DENSE_INDEX_NAME,
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": self._settings.milvus_dense_index_nlist},
        )
        self._client.create_index(
            collection_name=self._collection_name,
            index_params=indexes,
        )
        self._client.load_collection(collection_name=self._collection_name)
        self._logger.info(
            "Milvus dense index ready: collection=%s index=%s nlist=%s",
            self._collection_name,
            DENSE_INDEX_NAME,
            self._settings.milvus_dense_index_nlist,
        )


def main() -> None:
    """运行需显式提供 --yes，避免误触发索引重建。"""
    parser = argparse.ArgumentParser(description="Rebuild the Milvus dense IVF_FLAT index.")
    parser.add_argument("--yes", action="store_true", help="Confirm the index rebuild.")
    arguments = parser.parse_args()
    if not arguments.yes:
        parser.error("--yes is required because rebuilding an index affects collection availability")
    MilvusDenseIndexManager(settings).rebuild_dense_index()


if __name__ == "__main__":
    main()
