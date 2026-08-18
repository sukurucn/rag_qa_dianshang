from __future__ import annotations

from typing import Any

from dataprocess.milvus_index import MilvusDenseIndexManager
from dataprocess.milvus_store import COLLECTION_NAME, DENSE_INDEX_NAME
from tests.mysql_qa.testMysqlClient import make_settings


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self) -> None:
        self.params = FakeIndexParams()
        self.operations: list[tuple[str, str]] = []

    def has_collection(self, *, collection_name: str) -> bool:
        return True

    def list_indexes(self, *, collection_name: str, field_name: str) -> list[str]:
        assert field_name == "dense_vector"
        return ["dense_vector"]

    def release_collection(self, *, collection_name: str) -> None:
        self.operations.append(("release", collection_name))

    def drop_index(self, *, collection_name: str, index_name: str) -> None:
        self.operations.append(("drop", index_name))

    def prepare_index_params(self) -> FakeIndexParams:
        return self.params

    def create_index(self, *, collection_name: str, index_params: FakeIndexParams) -> None:
        assert index_params is self.params
        self.operations.append(("create", collection_name))

    def load_collection(self, *, collection_name: str) -> None:
        self.operations.append(("load", collection_name))


def testExplicitlyRebuildsDenseIndexAsIvfFlat() -> None:
    client = FakeMilvusClient()
    manager = MilvusDenseIndexManager(make_settings(), client=client)

    manager.rebuild_dense_index()

    assert client.operations == [
        ("release", COLLECTION_NAME),
        ("drop", "dense_vector"),
        ("create", COLLECTION_NAME),
        ("load", COLLECTION_NAME),
    ]
    assert client.params.indexes == [
        {
            "field_name": "dense_vector",
            "index_name": DENSE_INDEX_NAME,
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
    ]
