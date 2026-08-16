"""rag_qa 模块的领域模型与可替换依赖接口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dataprocess.models import Embedding


@dataclass(frozen=True)
class ParentChunk:
    """可作为回答上下文的完整父块。"""

    chunk_id: str
    source: str
    title_path: str
    text: str


@dataclass(frozen=True)
class WebSearchResult:
    """一条可展示、可供大模型补充参考的网页摘要。"""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class AnswerGeneration:
    """Agent 最终文本及其实际调用网络工具获得的摘要。"""

    text: str
    web_results: tuple[WebSearchResult, ...] = ()


@dataclass(frozen=True)
class RetrievalReport:
    """双路子块检索及父块去重后的观测数据。"""

    parents: tuple[ParentChunk, ...]
    dense_hit_count: int
    sparse_hit_count: int


@dataclass(frozen=True)
class RagQaResult:
    """RAG 回答和实际采用的父块。"""

    answer: str
    parents: tuple[ParentChunk, ...]
    fallback_reason: str | None = None
    web_results: tuple[WebSearchResult, ...] = ()


class Embedder(Protocol):
    """将改写查询编码为 BGE-M3 稠密和稀疏向量。"""

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """返回与输入顺序一致的向量。"""


class ParentRetriever(Protocol):
    """从向量库返回已去重的父块。"""

    def retrieve(self, queries: Sequence[str]) -> RetrievalReport:
        """检索各查询并通过子块关系找回父块。"""


class ParentReranker(Protocol):
    """按改写查询重排父块。"""

    def rerank(self, queries: Sequence[str], parents: Sequence[ParentChunk]) -> list[ParentChunk]:
        """返回按相关度降序排列的父块。"""


class AnswerModel(Protocol):
    """只根据指定父块生成可引用回答的模型。"""

    def answer(self, question: str, parents: Sequence[ParentChunk]) -> AnswerGeneration:
        """回答问题；无依据时返回约定的 UNANSWERABLE 标记。"""
