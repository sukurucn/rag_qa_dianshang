# rag_achivement_with_codex

面向电商问答场景的检索增强生成（RAG）知识库项目。系统将商品、订单、售后和运营知识接入可追溯的检索与问答流程，为用户提供基于知识库依据的回答。

## 技术栈

- Python 3.10
- LangChain、LangGraph：LLM 调用与工作流编排
- Milvus：向量存储和语义检索
- Redis：缓存与短期状态
- MySQL：关系型业务数据

## 规划架构

```text
config/       运行配置（敏感值仅通过环境变量或被忽略的本地文件提供）
src/          按功能划分的应用代码
  documentIngestion/  文档解析、清洗与入库
  retrieval/          查询改写、向量检索与重排
  answerGeneration/   基于上下文的回答生成与引用
  dataAccess/         Redis、Milvus、MySQL 适配层
tests/        与 src/ 模块一一对应的测试
```

## 开发原则

- 遵循 SOLID 原则，业务逻辑依赖接口，不直接耦合数据库或 LLM 客户端。
- 每项功能均需在 `src/` 和 `tests/` 中同步新增对应模块。
- 项目名称、模块、目录和自定义标识符使用驼峰命名法，例如 `vectorRetriever`。
- 不提交密码、API 密钥或真实生产数据。

## 分支与质量门禁

`master` 为受保护的稳定分支，`devlop` 为日常开发分支。功能分支从 `devlop` 创建；仅当功能完成、测试通过、代码审查完成并获人工批准后，才可合并至 `master`。

## 当前状态

项目骨架初始化中。依赖管理、可执行开发命令、测试框架和部署配置将在对应模块开始实现前确定并补充。
