"""从根目录 .env 加载应用基础设施配置。"""

import os
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """MySQL、Redis 与 Milvus 的已验证连接配置。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    mysql_host: str = Field(validation_alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, validation_alias="MYSQL_PORT", ge=1, le=65535)
    mysql_user: str = Field(validation_alias="MYSQL_USER")
    mysql_password: SecretStr = Field(validation_alias="MYSQL_PASSWORD")
    mysql_database: str = Field(validation_alias="MYSQL_DATABASE")
    mysql_charset: str = Field(default="utf8mb4", validation_alias="MYSQL_CHARSET")

    redis_host: str = Field(validation_alias="REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT", ge=1, le=65535)
    redis_db: int = Field(default=0, validation_alias="REDIS_DB", ge=0)
    redis_password: SecretStr = Field(validation_alias="REDIS_PASSWORD")

    mysql_qa_table_name: str = Field(
        default="question_answers", validation_alias="MYSQL_QA_TABLE_NAME"
    )
    mysql_qa_redis_key: str = Field(
        default="mysql_qa:questions", validation_alias="MYSQL_QA_REDIS_KEY"
    )
    mysql_qa_threshold: float = Field(
        default=0.8, validation_alias="MYSQL_QA_THRESHOLD", ge=0.0, le=1.0
    )

    rewrite_llm_base_url: str = Field(
        default="http://127.0.0.1:8000/v1", validation_alias="REWRITE_LLM_BASE_URL"
    )
    rewrite_llm_api_key: SecretStr | None = Field(
        default=None, validation_alias="REWRITE_LLM_API_KEY"
    )
    rewrite_llm_model: str = Field(default="gpt-4o-mini", validation_alias="REWRITE_LLM_MODEL")
    rewrite_llm_timeout_seconds: float = Field(
        default=30.0, validation_alias="REWRITE_LLM_TIMEOUT_SECONDS", gt=0
    )
    rewrite_llm_max_tokens: int = Field(
        default=512, validation_alias="REWRITE_LLM_MAX_TOKENS", ge=64, le=2048
    )
    rewrite_max_rounds: int = Field(
        default=3, validation_alias="REWRITE_MAX_ROUNDS", ge=1, le=3
    )
    rewrite_max_subquestions: int = Field(
        default=6, validation_alias="REWRITE_MAX_SUBQUESTIONS", ge=2, le=6
    )

    rag_top_k: int = Field(default=5, validation_alias="RAG_TOP_K", ge=1, le=20)
    rag_final_parent_count: int = Field(
        default=2, validation_alias="RAG_FINAL_PARENT_COUNT", ge=1, le=5
    )
    rag_customer_service_phone: str = Field(
        default="30129032", validation_alias="RAG_CUSTOMER_SERVICE_PHONE"
    )
    web_search_max_results: int = Field(
        default=5, validation_alias="WEB_SEARCH_MAX_RESULTS", ge=1, le=10
    )
    web_search_timeout_seconds: int = Field(
        default=10, validation_alias="WEB_SEARCH_TIMEOUT_SECONDS", ge=1, le=30
    )
    web_search_region: str = Field(default="cn-zh", validation_alias="WEB_SEARCH_REGION")
    milvus_dense_index_nlist: int = Field(
        default=128, validation_alias="MILVUS_DENSE_INDEX_NLIST", ge=1
    )
    milvus_dense_search_nprobe: int = Field(
        default=10, validation_alias="MILVUS_DENSE_SEARCH_NPROBE", ge=1
    )

    milvus_host: str = Field(validation_alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, validation_alias="MILVUS_PORT", ge=1, le=65535)
    milvus_user: str = Field(validation_alias="MILVUS_USER")
    milvus_password: SecretStr = Field(validation_alias="MILVUS_PASSWORD")
    milvus_database: str = Field(default="default", validation_alias="MILVUS_DATABASE")
    mineru_api_key: SecretStr | None = Field(default=None, validation_alias="MINERU_API_KEY")

    langsmith_tracing: bool = Field(default=True, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="rag-agentic-local", validation_alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str | None = Field(default=None, validation_alias="LANGSMITH_ENDPOINT")
    langsmith_workspace_id: str | None = Field(default=None, validation_alias="LANGSMITH_WORKSPACE_ID")

    @property
    def mysql_url(self) -> str:
        """返回供 SQLAlchemy/PyMySQL 使用的 MySQL 连接 URL。"""
        user = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password.get_secret_value())
        database = quote_plus(self.mysql_database)
        return (
            f"mysql+pymysql://{user}:{password}@{self.mysql_host}:{self.mysql_port}/"
            f"{database}?charset={self.mysql_charset}"
        )

    @property
    def redis_url(self) -> str:
        """返回供 redis-py 使用的 Redis 连接 URL。"""
        password = quote_plus(self.redis_password.get_secret_value())
        return f"redis://:{password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def milvus_uri(self) -> str:
        """返回 Milvus HTTP/gRPC 服务地址。"""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def milvus_token(self) -> str:
        """返回 pymilvus 使用的 user:password 认证令牌。"""
        return f"{self.milvus_user}:{self.milvus_password.get_secret_value()}"

    def configure_langsmith_tracing(self) -> bool:
        """在创建 LangChain 模型前，将可选的 LangSmith 配置写入进程环境。"""
        if not self.langsmith_tracing or self.langsmith_api_key is None:
            os.environ["LANGSMITH_TRACING"] = "false"
            return False

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        self._set_optional_env("LANGSMITH_ENDPOINT", self.langsmith_endpoint)
        self._set_optional_env("LANGSMITH_WORKSPACE_ID", self.langsmith_workspace_id)
        return True

    @staticmethod
    def _set_optional_env(name: str, value: str | None) -> None:
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)


# Pydantic Settings fills these required values from the root .env at runtime.
settings = Settings()  # type: ignore[call-arg]
