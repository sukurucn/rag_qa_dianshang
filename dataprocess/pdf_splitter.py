"""将超长 PDF 切成符合 MinerU 上传限制的临时文件。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader, PdfWriter

MAX_PDF_PAGES = 99


@dataclass(frozen=True)
class PdfPart:
    """一个待上传的 PDF 及其在原文件中的 1-based 页码范围。"""

    path: Path
    start_page: int
    end_page: int


def split_pdf(path: Path, output_directory: Path, max_pages: int = MAX_PDF_PAGES) -> list[PdfPart]:
    """仅在超过每片页数时创建分片；不修改原始 PDF。"""
    reader = PdfReader(path)
    page_count = len(reader.pages)
    if page_count <= max_pages:
        return [PdfPart(path=path, start_page=1, end_page=page_count)]

    output_directory.mkdir(parents=True, exist_ok=True)
    parts: list[PdfPart] = []
    for start_index in range(0, page_count, max_pages):
        writer = PdfWriter()
        end_index = min(start_index + max_pages, page_count)
        for page_index in range(start_index, end_index):
            writer.add_page(reader.pages[page_index])
        part_path = output_directory / f"{path.stem}.pages-{start_index + 1}-{end_index}.pdf"
        with part_path.open("wb") as stream:
            writer.write(stream)
        parts.append(PdfPart(path=part_path, start_page=start_index + 1, end_page=end_index))
    return parts
