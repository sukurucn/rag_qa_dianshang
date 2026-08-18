"""基于 Markdown 文档结构的父子分块逻辑。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from dataprocess.models import Chunk

CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 50
PARENT_CHILD_COUNT = 5
PARENT_CHILD_OVERLAP = 1
PARENT_CHUNK_SIZE = CHILD_CHUNK_SIZE * PARENT_CHILD_COUNT

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_DIVIDER_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?:\n\n|\n|[。！？!?；;])")


@dataclass(frozen=True)
class _Block:
    """解析后的最小结构单元。"""

    text: str
    title_path: str
    block_type: str


@dataclass(frozen=True)
class _ChildDraft:
    """尚未绑定父块的子块。"""

    chunk_id: str
    title_path: str
    block_type: str
    text: str


def normalize_text(text: str) -> str:
    """规范化换行和空白，确保全文 MD5 可重现。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def create_chunks(source: str, text: str) -> list[Chunk]:
    """按标题、段落、表格和代码结构创建带重叠的父子块。

    子块目标长度为 300 个 Unicode 字符，相邻块保留 50 字符重叠。每个
    父块由约五个连续子块组成；相邻父块共享一个子块。表格和代码块始终保持完整。
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    document_id = _md5(normalized)
    drafts = _create_child_drafts(document_id, _parse_blocks(normalized))
    if not drafts:
        return []

    parent_ids_by_child, parent_chunks = _create_parent_chunks(document_id, source, drafts)
    child_chunks = [
        Chunk(
            chunk_id=draft.chunk_id,
            document_id=document_id,
            parent_id=parent_ids[0],
            parent_ids=tuple(parent_ids),
            source=source,
            text=draft.text,
            chunk_type="child",
            title_path=draft.title_path,
            block_type=draft.block_type,
        )
        for draft, parent_ids in zip(drafts, parent_ids_by_child, strict=True)
    ]
    return [*parent_chunks, *child_chunks]


def _parse_blocks(markdown: str) -> list[_Block]:
    lines = markdown.splitlines()
    headings: list[str] = []
    blocks: list[_Block] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        heading_match = _HEADING_PATTERN.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            headings = [*headings[: level - 1], heading]
            index += 1
            continue

        title_path = " > ".join(headings) or "文档正文"
        if line.lstrip().startswith("```"):
            end_index = _consume_fenced_block(lines, index)
            blocks.append(_Block("\n".join(lines[index:end_index]), title_path, "code"))
            index = end_index
            continue

        if _is_table_start(lines, index):
            end_index = _consume_table(lines, index)
            blocks.append(_Block("\n".join(lines[index:end_index]), title_path, "table"))
            index = end_index
            continue

        end_index = _consume_paragraph(lines, index)
        paragraph = "\n".join(lines[index:end_index]).strip()
        if paragraph:
            blocks.append(_Block(paragraph, title_path, "text"))
        index = end_index
    return blocks


def _create_child_drafts(document_id: str, blocks: list[_Block]) -> list[_ChildDraft]:
    drafts: list[_ChildDraft] = []
    buffered_text: list[str] = []
    buffered_title_path: str | None = None

    def flush_text_buffer() -> None:
        nonlocal buffered_text, buffered_title_path
        if buffered_title_path is None:
            return
        for child_text in _split_text_with_overlap("\n\n".join(buffered_text)):
            drafts.append(_draft(document_id, len(drafts), buffered_title_path, "text", child_text))
        buffered_text = []
        buffered_title_path = None

    for block in blocks:
        if block.block_type == "text":
            if buffered_title_path not in {None, block.title_path}:
                flush_text_buffer()
            buffered_title_path = block.title_path
            buffered_text.append(block.text)
            continue

        flush_text_buffer()
        drafts.append(_draft(document_id, len(drafts), block.title_path, block.block_type, block.text))

    flush_text_buffer()
    return drafts


def _create_parent_chunks(
    document_id: str,
    source: str,
    drafts: list[_ChildDraft],
) -> tuple[list[list[str]], list[Chunk]]:
    parent_ids_by_child: list[list[str]] = [[] for _ in drafts]
    parent_chunks: list[Chunk] = []
    step = PARENT_CHILD_COUNT - PARENT_CHILD_OVERLAP

    starts = [0]
    while starts[-1] + PARENT_CHILD_COUNT < len(drafts):
        starts.append(starts[-1] + step)

    for parent_index, start in enumerate(starts):
        members = drafts[start : start + PARENT_CHILD_COUNT]
        member_ids = ",".join(member.chunk_id for member in members)
        parent_id = _md5(f"{document_id}:parent:{parent_index}:{member_ids}")
        for child_index in range(start, start + len(members)):
            parent_ids_by_child[child_index].append(parent_id)
        parent_chunks.append(
            Chunk(
                chunk_id=parent_id,
                document_id=document_id,
                parent_id=parent_id,
                parent_ids=(parent_id,),
                source=source,
                text="\n\n".join(member.text for member in members),
                chunk_type="parent",
                title_path=_common_title_path([member.title_path for member in members]),
                block_type="section",
            )
        )
    return parent_ids_by_child, parent_chunks


def _draft(
    document_id: str,
    index: int,
    title_path: str,
    block_type: str,
    text: str,
) -> _ChildDraft:
    child_id = _md5(f"{document_id}:child:{index}:{title_path}:{block_type}:{text}")
    return _ChildDraft(child_id, title_path, block_type, text)


def _split_text_with_overlap(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= CHILD_CHUNK_SIZE:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHILD_CHUNK_SIZE, len(text))
        if end < len(text):
            end = _find_boundary(text, start, end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - CHILD_CHUNK_OVERLAP)
    return chunks


def _find_boundary(text: str, start: int, hard_end: int) -> int:
    minimum_boundary = start + CHILD_CHUNK_SIZE // 2
    boundary_end: int | None = None
    for match in _SENTENCE_BOUNDARY_PATTERN.finditer(text, start, hard_end):
        if match.end() >= minimum_boundary:
            boundary_end = match.end()
    return boundary_end if boundary_end is not None else hard_end


def _consume_fenced_block(lines: list[str], start: int) -> int:
    index = start + 1
    while index < len(lines):
        if lines[index].lstrip().startswith("```"):
            return index + 1
        index += 1
    return len(lines)


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and _TABLE_DIVIDER_PATTERN.match(lines[index + 1]) is not None
    )


def _consume_table(lines: list[str], start: int) -> int:
    index = start + 2
    while index < len(lines) and lines[index].strip() and "|" in lines[index]:
        index += 1
    return index


def _consume_paragraph(lines: list[str], start: int) -> int:
    index = start
    while index < len(lines):
        if index > start and (
            not lines[index].strip()
            or _HEADING_PATTERN.match(lines[index])
            or lines[index].lstrip().startswith("```")
            or _is_table_start(lines, index)
        ):
            break
        index += 1
    return index


def _common_title_path(paths: list[str]) -> str:
    if not paths:
        return "文档正文"
    split_paths = [path.split(" > ") for path in paths]
    common: list[str] = []
    for parts in zip(*split_paths):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return " > ".join(common) or "文档正文"


def _md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
