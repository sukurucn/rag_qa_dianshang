# 更新日志

## Unreleased

- 将 DuckDuckGo 改为回答 Agent 在 Milvus 检索与重排之后可选调用的工具；通用知识不再在分类后直接联网，所有 RAG 问题均先读取本地上下文。本地 RAG 原文仍是最高事实优先级。
- 新增 Agent 工具调用循环和网页引用回传：仅模型实际请求 `search_web` 时才执行搜索，并由 `web_search_used`、`web_citations` 暴露实际结果；不支持 tool calling 的端点安全降级为本地 RAG。
- 新增持久化模块开关：FAQ（MySQL + Redis）与意图分类可在 Streamlit “模块控制”页切换，首次默认开启。
- 分类关闭时跳过分类和改写，原问题直接进入本地 RAG；FAQ 关闭时跳过 Redis 与 MySQL QA 查询。联网是否补充始终由回答 Agent 根据已检索的本地上下文决定。
- 明确 MySQL 使用 Docker 卷、Redis 使用 AOF 与 Docker 卷，均采用 `unless-stopped` 重启策略。
