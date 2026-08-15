"""严格解析 QA CSV 与 JSONL 导入文件。"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable


class QaImportValidationError(ValueError):
    """导入文件不满足 question/answer 固定字段约定。"""


def parse_qa_import(filename: str, payload: bytes) -> list[tuple[str, str]]:
    """按文件后缀解析 UTF-8 CSV 或 JSONL，并在写库前完成全部校验。"""
    suffix = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise QaImportValidationError("导入文件必须使用 UTF-8 编码") from error
    if suffix == "csv":
        rows: Iterable[object] = csv.DictReader(io.StringIO(text))
    elif suffix in {"jsonl", "json"}:
        rows = _jsonl_rows(text)
    else:
        raise QaImportValidationError("仅支持 CSV 或 JSONL 文件")
    parsed: list[tuple[str, str]] = []
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) != {"question", "answer"}:
            raise QaImportValidationError(
                f"第 {line_number} 条记录必须且只能包含 question、answer 字段"
            )
        question, answer = row["question"], row["answer"]
        if not isinstance(question, str) or not isinstance(answer, str):
            raise QaImportValidationError(f"第 {line_number} 条记录的 question 和 answer 必须是字符串")
        if not question.strip() or not answer.strip():
            raise QaImportValidationError(f"第 {line_number} 条记录的 question 和 answer 不能为空")
        parsed.append((question, answer))
    if not parsed:
        raise QaImportValidationError("导入文件没有有效的 QA 记录")
    return parsed


def _jsonl_rows(text: str) -> Iterable[object]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as error:
            raise QaImportValidationError(f"第 {line_number} 行不是有效 JSON") from error
