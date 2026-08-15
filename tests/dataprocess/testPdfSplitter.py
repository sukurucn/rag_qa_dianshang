from pathlib import Path

from pypdf import PdfReader, PdfWriter

from dataprocess.pdf_splitter import split_pdf


def testSplitsPdfIntoFilesStrictlyUnderOneHundredPages(tmp_path: Path) -> None:
    source = tmp_path / "catalog.pdf"
    writer = PdfWriter()
    for _ in range(100):
        writer.add_blank_page(width=72, height=72)
    with source.open("wb") as stream:
        writer.write(stream)

    parts = split_pdf(source, tmp_path / "parts")

    assert [(part.start_page, part.end_page) for part in parts] == [(1, 99), (100, 100)]
    assert [len(PdfReader(part.path).pages) for part in parts] == [99, 1]
    assert len(PdfReader(source).pages) == 100
