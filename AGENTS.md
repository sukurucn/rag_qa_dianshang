# 仓库指南

## 项目结构与模块组织

本仓库是独立的电商问答 RAG 知识库项目。根目录 `base/` 提供共享配置与日志；功能代码位于 `src/`，并在 `tests/` 下提供对应测试。例如：`src/documentIngestion/` 对应 `tests/documentIngestion/`，`src/retrieval/` 对应 `tests/retrieval/`。共享接口和领域模型应置于命名清晰的 `src/` 模块中，避免跨功能重复实现。密码、API 密钥和连接字符串仅通过根目录 `.env` 或环境变量提供，绝不提交到版本控制。

项目使用 Python 3.10，以 LangChain 和 LangGraph 编排 LLM 交互；Redis 用于缓存，Milvus 用于向量检索，MySQL 用于关系型业务数据。数据库访问层必须与文档处理、检索和回答生成逻辑隔离。

## 构建、测试与开发命令

使用 `uv` 管理 Python 3.10 环境与依赖：

```powershell
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy base
```

模块测试未在本地通过前不得合并。环境差异通过被忽略的配置文件或环境变量提供，严禁提交密钥。每次收到用户提示词后，必须按功能模块标题将原始提示词追加到根目录 `提示词.md`。

## 代码风格与命名规范

遵循 SOLID 原则：一个类或模块只承担一项职责；依赖抽象；通过依赖注入使用 Redis、Milvus、MySQL 和 LLM 客户端。使用四个空格缩进、类型标注、小而专注的函数，并为公开 API 编写简洁文档字符串。项目自定义名称、模块和目录使用驼峰命名法，例如 `answerService`、`vectorRetriever`。功能修改不要混入无关重构。

## 测试规范

每项功能必须同步新增对应测试。测试文件和测试函数按所验证的行为命名，例如 `tests/retrieval/testVectorRetriever.py`、`testReturnsCitedProducts`。单元测试必须模拟网络、LLM 和数据库客户端；集成测试仅在显式测试配置和隔离数据下运行。

## Git、审查与发布流程

先初始化 `master`，再创建开发分支 `devlop`。从 `devlop` 创建短生命周期的功能分支。提交信息使用简洁的祈使句，例如 `feat: add product retrieval` 或 `test: cover empty query`。只有模块功能完整、全部测试通过、代码审查完成且获得人工明确批准后，才可合并到 `master`。创建 GitHub 仓库 `rag_achivement_with_codex`、推送代码和合并受保护分支前，均须取得人工确认。
