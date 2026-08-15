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

## 本地知识库管理 API

先在本机启动 MySQL、Redis、Milvus，再运行：

```powershell
uv sync --locked --all-groups
uv run uvicorn admin_api.main:app --host 127.0.0.1 --port 8001
```

接口只允许监听 `127.0.0.1`，没有认证机制，不能改为对局域网或公网暴露。它提供 QA 的新增、导入、查询、删除，以及文档上传、异步入库、任务查询和删除。上传的原文件保存在被 Git 忽略的 `uploads/`；已完成文档删除时会清除其 Milvus 父子块和原文件，同时保留 MySQL 任务审计记录。

开发验证命令：

```powershell
uv run pytest
uv run ruff check .
uv run mypy base
```

## 用户问答 API

启动独立的用户问答入口：

```powershell
uv run uvicorn query_api.main:app --host 127.0.0.1 --port 8000
```

`POST /query` 会先检索 Redis + MySQL FAQ；FAQ 未命中后进行通用知识/专业咨询分类。
通用知识使用原问题直接 RAG，专业咨询先改写，再执行 BGE-M3 稠密和稀疏混合检索、父块回查、
BGE reranker 重排和带原文引用的回答。

新建 Milvus collection 使用 `IVF_FLAT` 稠密索引（`nlist=128`），检索使用 `nprobe=10`。
已有 collection 只在维护窗口执行以下显式命令进行切换：

```powershell
uv run python -m dataprocess.milvus_index --yes
```
