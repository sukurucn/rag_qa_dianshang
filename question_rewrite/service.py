"""专业咨询问题的最多三轮改写服务。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from base.config import Settings
from base.logger import get_logger
from question_rewrite.models import (
    RewriteModel,
    RewriteModelResponse,
    RewriteResult,
    RewriteStrategy,
    StrategyTrace,
)
from question_rewrite.prompt import REWRITE_SYSTEM_PROMPT

PROFESSIONAL_CONSULTATION = "专业咨询"


class QuestionRewriteService:
    """将专业咨询转为可直接路由到 RAG 的一组查询。"""

    def __init__(self, settings: Settings, model: RewriteModel) -> None:
        self._settings = settings
        self._model = model
        self._logger = get_logger("question_rewrite.service")

    def rewrite(
        self,
        user_question: str,
        classification: str,
        history: Sequence[BaseMessage] = (),
    ) -> RewriteResult:
        """执行最多三轮策略；输入 history 始终只读且不会被改写。"""
        if not user_question.strip():
            raise ValueError("user_question must not be empty")

        original_question = user_question.strip()
        if classification != PROFESSIONAL_CONSULTATION:
            return self._build_result(original_question, (original_question,), (), None)

        history_snapshot = tuple(history)
        current_question = original_question
        traces: list[StrategyTrace] = []
        for round_index in range(1, self._settings.rewrite_max_rounds + 1):
            messages = self._build_messages(history_snapshot, current_question)
            try:
                response = self._model.invoke(messages)
                trace = self._to_trace(round_index, response)
                traces.append(trace)
                self._logger.info(
                    "question rewrite round=%s strategy=%s input_tokens=%s output_tokens=%s",
                    round_index,
                    response.strategy.value,
                    response.input_tokens,
                    response.output_tokens,
                )
                outcome = self._apply_strategy(response, current_question)
            except Exception:
                self._logger.exception("Question rewrite failed at round=%s; using latest rewrite", round_index)
                return self._build_result(
                    original_question,
                    (current_question,),
                    tuple(traces),
                    "rewrite_error",
                )

            if outcome.terminal_queries is not None:
                return self._build_result(
                    original_question,
                    outcome.terminal_queries,
                    tuple(traces),
                    outcome.fallback_reason,
                )
            assert outcome.next_question is not None
            current_question = outcome.next_question

        self._logger.info("Question rewrite reached max rounds; using latest rewritten question")
        return self._build_result(original_question, (current_question,), tuple(traces), "max_rounds")

    def _build_messages(
        self, history: tuple[BaseMessage, ...], current_question: str
    ) -> list[BaseMessage]:
        """构造临时消息列表，不向调用方 history 中追加任何内容。"""
        return [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            *history,
            HumanMessage(content=f"当前需要改写的问题：{current_question}"),
        ]

    def _apply_strategy(
        self, response: RewriteModelResponse, current_question: str
    ) -> _StrategyOutcome:
        if response.strategy is RewriteStrategy.DIRECT:
            return _StrategyOutcome(terminal_queries=(current_question,))
        if response.strategy is RewriteStrategy.HYDE:
            if response.hypothetical_answer is None:
                raise ValueError("hyde strategy requires hypothetical_answer")
            return _StrategyOutcome(
                terminal_queries=self._unique_questions((current_question, response.hypothetical_answer))
            )
        if response.strategy is RewriteStrategy.SPLIT:
            questions = self._unique_questions(response.subquestions)
            if len(questions) < 2:
                raise ValueError("split strategy requires at least two subquestions")
            if len(questions) > self._settings.rewrite_max_subquestions:
                self._logger.warning(
                    "Split strategy returned too many subquestions: count=%s limit=%s",
                    len(questions),
                    self._settings.rewrite_max_subquestions,
                )
            return _StrategyOutcome(
                terminal_queries=questions[: self._settings.rewrite_max_subquestions],
                fallback_reason="subquestions_truncated"
                if len(questions) > self._settings.rewrite_max_subquestions
                else None,
            )
        if response.rewritten_question is None:
            raise ValueError("denoise strategy requires rewritten_question")
        if response.rewritten_question == current_question:
            return _StrategyOutcome(terminal_queries=(current_question,), fallback_reason="denoise_unchanged")
        return _StrategyOutcome(next_question=response.rewritten_question)

    @staticmethod
    def _unique_questions(questions: Sequence[str]) -> tuple[str, ...]:
        """移除空字符串和重复查询，并保持模型给出的顺序。"""
        return tuple(dict.fromkeys(question.strip() for question in questions if question.strip()))

    @staticmethod
    def _to_trace(round_index: int, response: RewriteModelResponse) -> StrategyTrace:
        return StrategyTrace(
            round_index=round_index,
            strategy=response.strategy,
            reason=response.reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        )

    @staticmethod
    def _build_result(
        original_question: str,
        rag_queries: tuple[str, ...],
        traces: tuple[StrategyTrace, ...],
        fallback_reason: str | None,
    ) -> RewriteResult:
        return RewriteResult(
            original_question=original_question,
            rag_queries=rag_queries,
            strategy_traces=traces,
            max_context_tokens=max((trace.input_tokens for trace in traces), default=0),
            total_input_tokens=sum(trace.input_tokens for trace in traces),
            total_output_tokens=sum(trace.output_tokens for trace in traces),
            fallback_reason=fallback_reason,
        )


class _StrategyOutcome:
    """策略执行后的内部控制流对象。"""

    def __init__(
        self,
        next_question: str | None = None,
        terminal_queries: tuple[str, ...] | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        self.next_question = next_question
        self.terminal_queries = terminal_queries
        self.fallback_reason = fallback_reason
