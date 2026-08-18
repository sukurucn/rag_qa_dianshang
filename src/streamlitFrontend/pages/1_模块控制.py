"""本机问答模块控制页。"""

from __future__ import annotations

import os

import streamlit as st

try:
    from src.streamlitFrontend.apiClient import (
        DEFAULT_QUERY_API_URL,
        FeatureFlags,
        QueryApiClient,
        QueryApiError,
    )
except ModuleNotFoundError:
    from apiClient import DEFAULT_QUERY_API_URL, FeatureFlags, QueryApiClient, QueryApiError


def main() -> None:
    """展示并持久化 FAQ 与意图分类开关。"""
    st.set_page_config(page_title="模块控制｜电商 RAG", page_icon="⚙️", layout="centered")
    st.title("模块控制")
    st.caption("控制会持久化保存；Query API 重启后仍生效。默认全部开启。")

    api_url = st.session_state.get("api_url") or os.getenv("RAG_QUERY_API_URL", DEFAULT_QUERY_API_URL)
    api_url = st.text_input("问答 API 地址", value=api_url).strip() or DEFAULT_QUERY_API_URL
    st.session_state.api_url = api_url
    client = QueryApiClient(api_url)
    try:
        flags = client.get_feature_flags()
    except QueryApiError as error:
        st.error(str(error))
        return
    finally:
        client.close()

    faq_enabled = st.toggle(
        "FAQ 模块（MySQL + Redis）",
        value=flags.faq_enabled,
        help="关闭后跳过 Redis 问题匹配和 MySQL 答案读取，直接进入后续问答路径。",
    )
    classifier_enabled = st.toggle(
        "意图分类模块",
        value=flags.classifier_enabled,
        help="关闭后不进行分类、问题改写或联网补充，原问题直接交给本地 RAG。",
    )
    if st.button("保存模块状态", type="primary", use_container_width=True):
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
                f"意图分类={'开启' if saved.classifier_enabled else '关闭'}。"
            )
        finally:
            client.close()

    st.divider()
    st.subheader("持久化基础设施")
    st.success("MySQL：Docker volume `mysql_data`，容器重启策略 `unless-stopped`。")
    st.success("Redis：AOF + Docker volume `redis_data`，容器重启策略 `unless-stopped`。")


if __name__ == "__main__":
    main()
