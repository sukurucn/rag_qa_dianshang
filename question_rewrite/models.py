"""问题改写模块的领域模型与可替换 LLM 协议。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from langchain_core.messages import BaseMessage


class RewriteStrategy(str, Enum):
    """改写模型可选择的受限策略。"""

    DENOISE = "denoise"
    HYDE = "hyde"
    SPLIT = "split"
    DIRECT = "direct"


@dataclass(frozen=True)
class RewriteModelResponse:
    """一次模型调用解析后的结构化结果。"""

    strategy: RewriteStrategy
    reason: str
    rewritten_question: str | None = None
    hypothetical_answer: str | None = None
    subquestions: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class RewriteModel(Protocol):
    """支持注入测试替身的 LangChain 模型适配接口。"""

    def invoke(self, messages: Sequence[BaseMessage]) -> RewriteModelResponse:
        """对一次完整提示消息返回结构化改写决策。"""


@dataclass(frozen=True)
class StrategyTrace:
    """一轮改写的可审计短轨迹，不包含模型完整思维链。"""

    round_index: int
    strategy: RewriteStrategy
    reason: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class RewriteResult:
    """交给后续 RAG 节点的查询及其改写观测信息。"""

    original_question: str
    rag_queries: tuple[str, ...]
    strategy_traces: tuple[StrategyTrace, ...]
    max_context_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    fallback_reason: str | None = None
