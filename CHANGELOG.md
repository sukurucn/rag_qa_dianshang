# 更新日志

## Unreleased

- 通用知识问答接入 DuckDuckGo 网络摘要与本地 RAG 联合回答；本地 RAG 原文被明确设为最高事实优先级。
- 新增持久化模块开关：FAQ（MySQL + Redis）与意图分类可在 Streamlit “模块控制”页切换，首次默认开启。
- 分类关闭时跳过分类、改写和联网补充，原问题直接进入本地 RAG；FAQ 关闭时跳过 Redis 与 MySQL QA 查询。
- 明确 MySQL 使用 Docker 卷、Redis 使用 AOF 与 Docker 卷，均采用 `unless-stopped` 重启策略。
