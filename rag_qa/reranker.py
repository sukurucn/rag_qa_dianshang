"""本地 BGE reranker-large 父块重排适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from base.logger import get_logger
from rag_qa.models import ParentChunk


class BgeParentReranker:
    """用本地 BGE reranker-large 对每个父块取多查询最高相关度。"""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path or Path(__file__).resolve().parent.parent / "models" / "bge-reranker-large"
        self._model: object | None = None
        self._logger = get_logger("rag_qa.reranker")

    def rerank(self, queries: Sequence[str], parents: Sequence[ParentChunk]) -> list[ParentChunk]:
        """使用所有改写/HyDE 查询对父块打分并取每块最高分。"""
        if not queries or not parents:
            return list(parents)
        model = cast(Any, self._get_model())
        scored_parents: list[tuple[float, int, ParentChunk]] = []
        for parent_index, parent in enumerate(parents):
            pairs = [[query, parent.text] for query in queries]
            scores = model.compute_score(pairs, normalize=True)
            score_values = scores if isinstance(scores, list) else [scores]
            scored_parents.append((max(float(score) for score in score_values), parent_index, parent))
        return [item[2] for item in sorted(scored_parents, key=lambda item: (-item[0], item[1]))]

    def _get_model(self) -> object:
        if self._model is None:
            if not self._model_path.is_dir():
                raise FileNotFoundError(f"BGE reranker model directory was not found: {self._model_path}")
            import torch
            from FlagEmbedding import FlagReranker  # type: ignore[import-untyped]

            use_cuda = torch.cuda.is_available()
            self._logger.info("Loading BGE reranker: cuda=%s", use_cuda)
            self._model = FlagReranker(
                str(self._model_path),
                use_fp16=use_cuda,
                devices=["cuda:0"] if use_cuda else None,
            )
        return self._model
