"""仅绑定本机的 Streamlit RAG 对话入口。"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

try:
    from src.streamlitFrontend.apiClient import (
        DEFAULT_QUERY_API_URL,
        FeatureFlags,
        QueryApiClient,
        QueryApiError,
    )
except ModuleNotFoundError:
    # Streamlit 直接执行本文件时仅将当前目录加入导入路径。
    from apiClient import DEFAULT_QUERY_API_URL, FeatureFlags, QueryApiClient, QueryApiError

st.set_page_config(page_title="电商 RAG 问答", page_icon="💬", layout="centered")


def main() -> None:
    """渲染会话内聊天记录，并调用本机问答 API。"""
    _initialize_session_state()
    st.title("电商 RAG 问答")
    st.caption("FAQ 优先；未命中时进入分类、检索、重排与引用回答。")

    with st.sidebar:
        st.header("连接设置")
        api_url = st.text_input("问答 API 地址", value=st.session_state.api_url)
        st.session_state.api_url = api_url.strip() or DEFAULT_QUERY_API_URL
        if st.button("检查后端连接", use_container_width=True):
            _check_connection(st.session_state.api_url)
        if st.button("清空本次会话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        _render_feature_controls(st.session_state.api_url)
        st.page_link("pages/1_模块控制.py", label="打开完整模块控制页", icon="⚙️")
        st.divider()
        st.caption("界面和 API 均只建议绑定到 127.0.0.1。")

    for message in st.session_state.messages:
        _render_message(message)

    if question := st.chat_input("请输入商品、订单、售后或运营相关问题"):
        _render_and_store_answer(question, st.session_state.api_url)


def _initialize_session_state() -> None:
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("RAG_QUERY_API_URL", DEFAULT_QUERY_API_URL)
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _check_connection(api_url: str) -> None:
    client = QueryApiClient(api_url)
    try:
        client.check_connection()
    except QueryApiError as error:
        st.error(str(error))
    else:
        st.success("问答 API 可连接。")
    finally:
        client.close()


def _render_feature_controls(api_url: str) -> None:
    """在首页侧栏直接提供 FAQ 与意图分类的持久化开关。"""
    st.subheader("模块状态")
    client = QueryApiClient(api_url)
    try:
        flags = client.get_feature_flags()
    except QueryApiError as error:
        st.warning(f"无法读取模块状态：{error}")
        return
    finally:
        client.close()

    faq_enabled = st.toggle(
        "FAQ（MySQL + Redis）",
        value=flags.faq_enabled,
        help="关闭后跳过 Redis 匹配和 MySQL FAQ，直接进入后续问答路径。",
    )
    classifier_enabled = st.toggle(
        "意图分类",
        value=flags.classifier_enabled,
        help="关闭后跳过分类、改写和联网补充，原问题直接进入 RAG。",
    )
    if st.button("保存模块状态", use_container_width=True):
        client = QueryApiClient(api_url)
        try:
            saved = client.update_feature_flags(
                FeatureFlags(faq_enabled=faq_enabled, classifier_enabled=classifier_enabled)
            )
        except QueryApiError as error:
            st.error(str(error))
        else:
            st.success(
                "已保存："
                f"FAQ={'开启' if saved.faq_enabled else '关闭'}，"
                f"分类={'开启' if saved.classifier_enabled else '关闭'}。"
            )
        finally:
            client.close()


def _render_and_store_answer(question: str, api_url: str) -> None:
    user_message: dict[str, Any] = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("正在查询知识库……"):
            client = QueryApiClient(api_url)
            try:
                answer = client.ask(question)
            except (QueryApiError, ValueError) as error:
                st.error(str(error))
                return
            finally:
                client.close()
        assistant_message = {
            "role": "assistant",
            "content": answer.answer,
            "source": answer.source,
            "classification": answer.classification,
            "faq_confidence": answer.faq_confidence,
            "classification_confidence": answer.classification_confidence,
            "fallback_reason": answer.fallback_reason,
            "citations": answer.citations,
            "web_citations": answer.web_citations,
            "web_search_used": answer.web_search_used,
        }
        _render_answer_details(assistant_message)
        st.session_state.messages.append(assistant_message)


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_answer_details(message)
        else:
            st.markdown(message["content"])


def _render_answer_details(message: dict[str, Any]) -> None:
    st.markdown(message["content"])
    source = "FAQ" if message["source"] == "faq" else "RAG"
    st.caption(f"来源：{source}｜FAQ 置信度：{message['faq_confidence']:.2%}")
    if message["classification"] is not None:
        confidence = message["classification_confidence"]
        suffix = f"（{confidence:.2%}）" if confidence is not None else ""
        st.caption(f"分类：{message['classification']}{suffix}")
    if message["fallback_reason"]:
        st.info(f"兜底原因：{message['fallback_reason']}")
    citations = message["citations"]
    if citations:
        with st.expander(f"引用原文（{len(citations)}）"):
            for index, citation in enumerate(citations, start=1):
                st.markdown(f"**{index}. {citation.title_path}**")
                st.caption(f"来源文件：{citation.source}｜块 ID：{citation.chunk_id}")
                st.write(citation.text)
    if message.get("web_search_used"):
        web_citations = message.get("web_citations", ())
        with st.expander(f"网络补充（{len(web_citations)}）"):
            if not web_citations:
                st.caption("已尝试 DuckDuckGo 检索，但没有可用的网页摘要；本次回答仅依赖本地 RAG。")
            for index, citation in enumerate(web_citations, start=1):
                st.markdown(f"**{index}. [{citation.title}]({citation.url})**")
                st.write(citation.snippet)


if __name__ == "__main__":
    main()
