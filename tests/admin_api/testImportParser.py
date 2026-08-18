from __future__ import annotations

import pytest

from admin_api.import_parser import QaImportValidationError, parse_qa_import


def testParsesCsvAndJsonl() -> None:
    assert parse_qa_import("qa.csv", b"question,answer\nQ1,A1\n") == [("Q1", "A1")]
    assert parse_qa_import("qa.jsonl", b'{"question":"Q2","answer":"A2"}\n') == [("Q2", "A2")]


def testRejectsUnexpectedImportFields() -> None:
    with pytest.raises(QaImportValidationError, match="question、answer"):
        parse_qa_import("qa.csv", b"question,answer,extra\nQ,A,x\n")
