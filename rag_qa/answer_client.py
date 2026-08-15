"""基于父块原文生成带引用回答的 LangChain 模型适配器。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from base.config import Settings
from rag_qa.models import ParentChunk

ANSWER_SYSTEM_PROMPT = """你是严谨的 IT 教育培训知识库问答助手。
只能根据给定的原文上下文回答，不得补充上下文中没有的信息。
请自行决定最自然的引用呈现方式，但回答中必须清晰指出所依据的原文来源、章节或原文片段。
如果上下文无法充分回答问题，只输出：UNANSWERABLE"""


class LangChainAnswerModel:
    """使用 OpenAI 兼容模型根据父块上下文生成最终回答。"""

    def __init__(self, settings: Settings, chat_model: BaseChatModel | None = None) -> None:
        self._chat_model = chat_model or ChatOpenAI(
            base_url=settings.rewrite_llm_base_url,
            api_key=settings.rewrite_llm_api_key,
            model=settings.rewrite_llm_model,
            temperature=0,
            timeout=settings.rewrite_llm_timeout_seconds,
            max_completion_tokens=settings.rewrite_llm_max_tokens,
        )

    def answer(self, question: str, parents: Sequence[ParentChunk]) -> str:
        """将问题和可追溯父块作为独立消息发送给回答模型。"""
        context = "\n\n".join(
            f"[父块 {index}] 来源：{parent.source}\n章节：{parent.title_path}\n原文：{parent.text}"
            for index, parent in enumerate(parents, start=1)
        )
        response = self._chat_model.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=f"问题：{question}\n\n原文上下文：\n{context}"),
            ]
        )
        if not isinstance(response, AIMessage) or not isinstance(response.content, str):
            raise TypeError("answer LLM must return textual AIMessage content")
        return response.content.strip()
