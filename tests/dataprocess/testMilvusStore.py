from __future__ import annotations

from base.config import Settings
from dataprocess.milvus_store import (
    COLLECTION_NAME,
    DENSE_INDEX_NAME,
    SPARSE_INDEX_NAME,
    MilvusDocumentStore,
)


class FakeMilvusClient:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def has_collection(self, *, collection_name: str) -> bool:
        assert collection_name == COLLECTION_NAME
        return True

    def delete(self, *, collection_name: str, filter: str) -> None:
        self.deleted.append((collection_name, filter))


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, object]] = []

    def add_index(self, **kwargs: object) -> None:
        self.indexes.append(kwargs)


class FakeCollectionClient:
    def __init__(self) -> None:
        self.indexes = FakeIndexParams()
        self.created_with: FakeIndexParams | None = None

    def has_collection(self, *, collection_name: str) -> bool:
        return False

    def prepare_index_params(self) -> FakeIndexParams:
        return self.indexes

    def create_collection(self, **kwargs: object) -> None:
        self.created_with = kwargs["index_params"]  # type: ignore[assignment]


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


def testDeletesOnlyExistingCollectionChunksForDocumentId() -> None:
    client = FakeMilvusClient()
    store = MilvusDocumentStore(make_settings(), client=client)

    store.delete_document("a" * 32)

    assert client.deleted == [(COLLECTION_NAME, 'document_id == "' + "a" * 32 + '"')]


def testCreatesIvfFlatAndSparseIndexesForNewCollection() -> None:
    client = FakeCollectionClient()
    store = MilvusDocumentStore(make_settings(), client=client)

    store._ensure_collection()

    assert client.created_with is client.indexes
    dense, sparse = client.indexes.indexes
    assert dense == {
        "field_name": "dense_vector",
        "index_name": DENSE_INDEX_NAME,
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    assert sparse["index_name"] == SPARSE_INDEX_NAME
    assert sparse["index_type"] == "SPARSE_INVERTED_INDEX"
