import os

import pytest
from pydantic import ValidationError

from base.config import Settings


def testBuildsDatabaseConnectionInformation() -> None:
    settings = Settings(
        _env_file=None,
        mysql_host="127.0.0.1",
        mysql_user="rag_user",
        mysql_password="My Sql+Password",
        mysql_database="rag_db",
        redis_host="127.0.0.1",
        redis_password="Redis Password",
        milvus_host="127.0.0.1",
        milvus_user="root",
        milvus_password="Milvus Password",
    )

    assert "My+Sql%2BPassword" in settings.mysql_url
    assert settings.redis_url == "redis://:Redis+Password@127.0.0.1:6379/0"
    assert settings.milvus_uri == "http://127.0.0.1:19530"
    assert settings.milvus_token == "root:Milvus Password"


def testRejectsMissingRequiredDatabaseSettings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MYSQL_HOST",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_DATABASE",
        "REDIS_HOST",
        "REDIS_PASSWORD",
        "MILVUS_HOST",
        "MILVUS_USER",
        "MILVUS_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def testConfiguresLangsmithTracingOnlyWhenAKeyIsAvailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    settings = Settings(
        _env_file=None,
        mysql_host="127.0.0.1",
        mysql_user="rag_user",
        mysql_password="password",
        mysql_database="rag_db",
        redis_host="127.0.0.1",
        redis_password="password",
        milvus_host="127.0.0.1",
        milvus_user="root",
        milvus_password="password",
        langsmith_api_key="langsmith-test-key",
        langsmith_project="rag-test",
    )
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGSMITH_WORKSPACE_ID", raising=False)

    assert settings.configure_langsmith_tracing() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "langsmith-test-key"
    assert os.environ["LANGSMITH_PROJECT"] == "rag-test"


def testDisablesLangsmithTracingWhenNoKeyIsConfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        _env_file=None,
        mysql_host="127.0.0.1",
        mysql_user="rag_user",
        mysql_password="password",
        mysql_database="rag_db",
        redis_host="127.0.0.1",
        redis_password="password",
        milvus_host="127.0.0.1",
        milvus_user="root",
        milvus_password="password",
        langsmith_api_key=None,
    )
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    assert settings.configure_langsmith_tracing() is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
