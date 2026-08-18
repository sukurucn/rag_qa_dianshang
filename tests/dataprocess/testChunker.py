from dataprocess.chunker import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHILD_COUNT,
    create_chunks,
)


def testCreatesStructureAwareParentsWithChildAndParentOverlap() -> None:
    text = "# Policy\n\n## Returns\n\n" + "a" * 1700

    chunks = create_chunks("D:/documents/product.md", text)
    parent_chunks = [chunk for chunk in chunks if chunk.chunk_type == "parent"]
    child_chunks = [chunk for chunk in chunks if chunk.chunk_type == "child"]

    assert len(child_chunks) == 7
    assert len(parent_chunks) == 2
    assert all(len(chunk.text) <= CHILD_CHUNK_SIZE for chunk in child_chunks)
    assert all(chunk.title_path == "Policy > Returns" for chunk in child_chunks)
    assert child_chunks[0].text[-CHILD_CHUNK_OVERLAP:] == child_chunks[1].text[:CHILD_CHUNK_OVERLAP]
    assert len(parent_chunks[0].text) >= len(child_chunks[0].text) * PARENT_CHILD_COUNT
    assert sum(parent_chunks[0].chunk_id in child.parent_ids for child in child_chunks) == 5
    assert sum(parent_chunks[1].chunk_id in child.parent_ids for child in child_chunks) == 3
    assert set(child_chunks[4].parent_ids) == {parent.chunk_id for parent in parent_chunks}
    assert all(len(chunk.chunk_id) == 32 for chunk in chunks)
    assert all(chunk.source == "D:/documents/product.md" for chunk in chunks)


def testPreservesTableAsAtomicStructuredChild() -> None:
    text = """# Product

## Specification

| Field | Value |
| --- | --- |
| Color | Blue |
| Warranty | One year |
"""

    chunks = create_chunks("source", text)
    table = next(chunk for chunk in chunks if chunk.chunk_type == "child")

    assert table.block_type == "table"
    assert table.title_path == "Product > Specification"
    assert "| Warranty | One year |" in table.text
    assert "章节: Product > Specification" in table.embedding_text


def testCreatesSameIdsForSameNormalizedContent() -> None:
    first = create_chunks("source", "first line\r\n\r\nsecond line")
    second = create_chunks("source", "first line\n\nsecond line")

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
