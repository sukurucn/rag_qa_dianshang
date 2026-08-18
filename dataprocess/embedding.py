"""本地 BGE-M3 向量生成适配器。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from base.logger import get_logger
from dataprocess.models import Embedding


class EmbeddingProvider(Protocol):
    """可替换的向量生成接口。"""

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """返回与输入文本一一对应的稠密和稀疏向量。"""


class BgeM3Embedder:
    """使用本地 models/bge-m3 生成 1024 维 dense 和 lexical sparse 向量。"""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = model_path or Path(__file__).resolve().parent.parent / "models" / "bge-m3"
        self._model: object | None = None

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """批量生成 BGE-M3 embedding，延迟加载模型以缩短普通导入时间。"""
        if not texts:
            return []
        model = self._get_model()
        encoded = model.encode(list(texts), return_dense=True, return_sparse=True)
        dense_vectors = encoded["dense_vecs"]
        sparse_vectors = encoded["lexical_weights"]
        return [
            Embedding(
                dense_vector=[float(value) for value in dense_vector],
                sparse_vector={int(key): float(value) for key, value in sparse_vector.items()},
            )
            for dense_vector, sparse_vector in zip(dense_vectors, sparse_vectors, strict=True)
        ]

    def _get_model(self) -> object:
        if self._model is None:
            if not self._model_path.is_dir():
                raise FileNotFoundError(f"BGE-M3 model directory was not found: {self._model_path}")
            import torch
            from FlagEmbedding import BGEM3FlagModel

            use_cuda = torch.cuda.is_available()
            logger = get_logger("dataprocess.embedding")
            if use_cuda:
                logger.info("Loading BGE-M3 on CUDA device 0 with FP16 inference.")
                self._model = BGEM3FlagModel(
                    str(self._model_path),
                    use_fp16=True,
                    devices=["cuda:0"],
                )
            else:
                logger.warning("CUDA is unavailable; loading BGE-M3 on CPU with FP32 inference.")
                self._model = BGEM3FlagModel(str(self._model_path), use_fp16=False)
        return self._model
