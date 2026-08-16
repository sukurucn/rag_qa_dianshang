"""LangChain OpenAI 兼容模型的结构化问题改写适配器。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from base.config import Settings
from question_rewrite.models import RewriteModelResponse, RewriteStrategy


class LangChainRewriteModel:
    """将 ChatOpenAI 响应解析成改写模块的稳定数据模型。"""

    def __init__(self, settings: Settings, chat_model: BaseChatModel | None = None) -> None:
        self._settings = settings
        self._chat_model = chat_model or self._create_chat_model()

    @traceable(name="question_rewrite_llm", run_type="chain")
    def invoke(self, messages: Sequence[BaseMessage]) -> RewriteModelResponse:
        """调用模型，并提取结构化策略和模型返回的 token 用量。"""
        response = self._chat_model.invoke(list(messages))
        if not isinstance(response, AIMessage):
            raise TypeError("rewrite LLM must return an AIMessage")
        content = response.content
        if not isinstance(content, str):
            raise TypeError("rewrite LLM response content must be a JSON string")
        payload = self._parse_json(content)
        usage = cast(Mapping[str, Any], response.usage_metadata or {})
        input_tokens = self._read_usage(usage, "input_tokens", "prompt_tokens")
        output_tokens = self._read_usage(usage, "output_tokens", "completion_tokens")
        total_tokens = self._read_usage(usage, "total_tokens")
        if input_tokens == 0:
            input_tokens = self._chat_model.get_num_tokens_from_messages(list(messages))
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        return RewriteModelResponse(
            strategy=RewriteStrategy(payload["strategy"]),
            reason=self._as_text(payload, "reason")[:40],
            rewritten_question=self._optional_text(payload, "rewritten_question"),
            hypothetical_answer=self._optional_text(payload, "hypothetical_answer"),
            subquestions=tuple(self._as_text_list(payload, "subquestions")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _create_chat_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self._settings.rewrite_llm_base_url,
            api_key=self._settings.rewrite_llm_api_key,
            model=self._settings.rewrite_llm_model,
            temperature=0,
            timeout=self._settings.rewrite_llm_timeout_seconds,
            max_completion_tokens=self._settings.rewrite_llm_max_tokens,
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            payload: object = json.loads(content)
        except json.JSONDecodeError as exception:
            raise ValueError("rewrite LLM returned invalid JSON") from exception
        if not isinstance(payload, dict):
            raise TypeError("rewrite LLM JSON response must be an object")
        return payload

    @staticmethod
    def _as_text(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"rewrite LLM field '{key}' must be a non-empty string")
        return value.strip()

    @staticmethod
    def _optional_text(payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"rewrite LLM field '{key}' must be a string")
        stripped_value = value.strip()
        return stripped_value or None

    @staticmethod
    def _as_text_list(payload: dict[str, Any], key: str) -> list[str]:
        value = payload.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise TypeError(f"rewrite LLM field '{key}' must be a string array")
        return [item.strip() for item in value if item.strip()]

    @staticmethod
    def _read_usage(usage: Mapping[str, Any], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                return value
        return 0
