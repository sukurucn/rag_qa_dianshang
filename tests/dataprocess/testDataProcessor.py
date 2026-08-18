from collections.abc import Sequence
from pathlib import Path

from dataprocess.data_processor import data_processor
from dataprocess.models import Chunk, Embedding


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return [Embedding(dense_vector=[0.0] * 1024, sparse_vector={1: 0.5}) for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings: list[Embedding] = []

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)


def testProcessesTextAndWritesParentChildChunks(tmp_path: Path) -> None:
    source = tmp_path / "product.txt"
    source.write_text("电商知识" * 200, encoding="utf-8")
    store = FakeStore()

    report = data_processor(
        source,
        start_infrastructure=False,
        embedder=FakeEmbedder(),
        store=store,
    )

    assert report.processed_documents == 1
    assert report.stored_chunks == len(store.chunks)
    assert any(chunk.chunk_type == "parent" for chunk in store.chunks)
    assert any(chunk.chunk_type == "child" for chunk in store.chunks)
    assert len(store.chunks) == len(store.embeddings)
