"""将 FAQ 未回答问题路由为通用知识或专业咨询。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from base.logger import get_logger
from model_trian_classify.data import ID_TO_LABEL, LABEL_TO_ID


class RouteLabel(str, Enum):
    """路由分类标签。"""

    GENERAL_KNOWLEDGE = "通用知识"
    PROFESSIONAL_CONSULTATION = "专业咨询"


@dataclass(frozen=True)
class RouteDecision:
    """可由上层 LangGraph 消费的路由结果。"""

    label: RouteLabel
    confidence: float
    target_route: str
    model_version: str


class QueryPredictor(Protocol):
    """可替换的二分类推理接口。"""

    def predict(self, question: str) -> tuple[int, float]:
        """返回标签 ID 与该标签概率。"""


class HuggingFaceQueryPredictor:
    """懒加载已蒸馏模型的本地 Hugging Face 推理适配器。"""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._device: object | None = None

    def predict(self, question: str) -> tuple[int, float]:
        """对单条问题推理并返回最高类别概率。"""
        self._ensure_loaded()
        import torch

        tokenizer = self._tokenizer
        model = self._model
        device = self._device
        encoded = tokenizer(question, return_tensors="pt", truncation=True, max_length=128)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        label_id = int(torch.argmax(probabilities).item())
        return label_id, float(probabilities[label_id].item())

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self._model_path.is_dir():
            raise FileNotFoundError(f"Query-router model directory was not found: {self._model_path}")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path,
            local_files_only=True,
        ).to(self._device)
        self._model.eval()


class QueryRouter:
    """分类 FAQ 未命中的问题；两类问题均由统一 RAG 流程回答。"""

    def __init__(
        self,
        predictor: QueryPredictor | None = None,
        *,
        model_path: Path | None = None,
        confidence_threshold: float = 0.7,
        model_version: str = "best_model",
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        default_model_path = Path(__file__).resolve().parent / "model" / "best_model"
        self._predictor = predictor or HuggingFaceQueryPredictor(model_path or default_model_path)
        self._confidence_threshold = confidence_threshold
        self._model_version = model_version
        self._logger = get_logger("model_trian_classify.router")

    def route(self, question: str) -> RouteDecision:
        """路由问题；不确定时保守地交由知识库检索。"""
        if not question.strip():
            return RouteDecision(
                label=RouteLabel.PROFESSIONAL_CONSULTATION,
                confidence=0.0,
                target_route="rag_qa",
                model_version=self._model_version,
            )
        label_id, confidence = self._predictor.predict(question)
        if label_id not in ID_TO_LABEL:
            raise RuntimeError(f"The model returned unsupported label id: {label_id}")
        label = RouteLabel(ID_TO_LABEL[label_id])
        target_route = "rag_qa"
        self._logger.info(
            "query router: label=%s confidence=%.4f target=%s",
            label.value,
            confidence,
            target_route,
        )
        return RouteDecision(label, confidence, target_route, self._model_version)


def label_id_for(route_label: RouteLabel) -> int:
    """提供路由标签到训练标签 ID 的显式映射。"""
    return LABEL_TO_ID[route_label.value]
