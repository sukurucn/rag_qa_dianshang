"""基于 BM25 内积（IP）的问题匹配。"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from mysql_qa.models import QuestionAnswer

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)


@dataclass(frozen=True)
class Match:
    """一个候选问题的原始 BM25-IP 分数和 softmax 置信度。"""

    question: QuestionAnswer
    score: float
    confidence: float


class Bm25InnerProductMatcher:
    """以 BM25 词权重向量的内积（IP）计算词法相似度。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b

    def match(self, query: str, candidates: Sequence[QuestionAnswer]) -> list[Match]:
        """返回按分数降序排列、且已 softmax 归一化的候选结果。"""
        query_terms = self._tokenize(query)
        if not query_terms or not candidates:
            return []

        documents = [self._tokenize(candidate.question) for candidate in candidates]
        document_frequency = Counter(term for document in documents for term in set(document))
        average_length = sum(len(document) for document in documents) / len(documents)
        query_counts = Counter(query_terms)
        scores = [
            self._inner_product(query_counts, document, document_frequency, len(documents), average_length)
            for document in documents
        ]
        confidences = self._softmax(scores)
        matches = [
            Match(question=candidate, score=score, confidence=confidence)
            for candidate, score, confidence in zip(candidates, scores, confidences, strict=True)
        ]
        return sorted(matches, key=lambda item: item.score, reverse=True)

    def _inner_product(
        self,
        query_counts: Counter[str],
        document: list[str],
        document_frequency: Counter[str],
        document_count: int,
        average_length: float,
    ) -> float:
        document_counts = Counter(document)
        score = 0.0
        for term, query_term_count in query_counts.items():
            frequency = document_counts[term]
            if frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            document_weight = frequency * (self._k1 + 1) / (
                frequency + self._k1 * (1 - self._b + self._b * len(document) / average_length)
            )
            score += query_term_count * inverse_document_frequency * document_weight
        return score

    @staticmethod
    def _softmax(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        maximum = max(scores)
        exponentials = [math.exp(score - maximum) for score in scores]
        denominator = sum(exponentials)
        return [value / denominator for value in exponentials]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())
