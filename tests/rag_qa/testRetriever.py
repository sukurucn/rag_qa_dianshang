from __future__ import annotations

from typing import Any

from dataprocess.models import Embedding
from rag_qa.retriever import CHILD_FILTER, MilvusHybridRetriever
from tests.mysql_qa.testMysqlClient import make_settings


class FakeEmbedder:
    def embed(self, texts: list[str] | tuple[str, ...]) -> list[Embedding]:
        return [Embedding(dense_vector=[0.1, 0.2], sparse_vector={3: 0.7}) for _ in texts]


class FakeMilvusClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self.get_ids: list[str] = []

    def search(self, **kwargs: Any) -> list[list[dict[str, str]]]:
        self.search_calls.append(kwargs)
        if kwargs["anns_field"] == "dense_vector":
            return [[{"chunk_id": "child-1", "parent_id": "parent-1"}, {"chunk_id": "child-2", "parent_id": "parent-2"}]]
        return [[{"chunk_id": "child-2", "parent_id": "parent-2"}, {"chunk_id": "child-3", "parent_id": "parent-3"}]]

    def get(self, **kwargs: Any) -> list[dict[str, str]]:
        self.get_ids = kwargs["ids"]
        return [
            {"chunk_id": "parent-1", "chunk_type": "parent", "source": "a.md", "title_path": "A", "text": "父块 A"},
            {"chunk_id": "parent-2", "chunk_type": "parent", "source": "b.md", "title_path": "B", "text": "父块 B"},
            {"chunk_id": "parent-2", "chunk_type": "parent", "source": "b.md", "title_path": "B", "text": "重复父块"},
            {"chunk_id": "parent-3", "chunk_type": "child", "source": "c.md", "title_path": "C", "text": "非父块"},
        ]


def testSearchesDenseAndSparseChildrenThenDeduplicatesParents() -> None:
    client = FakeMilvusClient()
    retriever = MilvusHybridRetriever(make_settings(), embedder=FakeEmbedder(), client=client)

    report = retriever.retrieve(("AI课程费用",))

    assert [call["anns_field"] for call in client.search_calls] == ["dense_vector", "sparse_vector"]
    assert [call["search_params"]["metric_type"] for call in client.search_calls] == ["COSINE", "IP"]
    assert [call["search_params"]["params"] for call in client.search_calls] == [
        {"nprobe": 10},
        {},
    ]
    assert all(call["limit"] == 5 and call["filter"] == CHILD_FILTER for call in client.search_calls)
    assert client.get_ids == ["parent-1", "parent-2", "parent-3"]
    assert [parent.chunk_id for parent in report.parents] == ["parent-1", "parent-2"]
    assert report.dense_hit_count == 2
    assert report.sparse_hit_count == 2
