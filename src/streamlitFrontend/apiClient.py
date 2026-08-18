"""Streamlit 前端访问本地问答 API 的可测试客户端。"""
# ruff: noqa: N999

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_QUERY_API_URL = "http://127.0.0.1:8000"


class QueryApiError(RuntimeError):
    """问答 API 不可用或返回不符合契约的数据时抛出。"""


@dataclass(frozen=True)
class Citation:
    """回答中携带的一条父块引用。"""

    chunk_id: str
    source: str
    title_path: str
    text: str


@dataclass(frozen=True)
class WebCitation:
    """供前端展示的网络检索摘要。"""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class FeatureFlags:
    """问答模块的当前启用状态。"""

    faq_enabled: bool
    classifier_enabled: bool


@dataclass(frozen=True)
class QueryResponse:
    """前端需要展示的标准化问答结果。"""

    answer: str
    source: str
    classification: str | None
    faq_confidence: float
    classification_confidence: float | None
    fallback_reason: str | None
    citations: tuple[Citation, ...]
    web_citations: tuple[WebCitation, ...]
    web_search_used: bool


class QueryApiClient:
    """通过 HTTP 调用仅绑定本机的 `query_api` 服务。"""

    def __init__(
        self,
        base_url: str = DEFAULT_QUERY_API_URL,
        *,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.Client()
        self._owns_client = client is None

    def ask(self, question: str) -> QueryResponse:
        """提交一个非空问题，并将后端响应转换为前端领域模型。"""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空")
        try:
            response = self._client.post(
                f"{self._base_url}/query",
                json={"question": normalized_question},
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise QueryApiError("问答服务响应超时，请稍后重试。") from error
        except httpx.ConnectError as error:
            raise QueryApiError("无法连接问答服务，请确认 query_api 已启动。") from error
        except httpx.HTTPStatusError as error:
            raise QueryApiError(_error_detail(error.response)) from error
        except httpx.HTTPError as error:
            raise QueryApiError("调用问答服务失败，请检查服务日志。") from error

        try:
            return _parse_response(response.json())
        except (KeyError, TypeError, ValueError) as error:
            raise QueryApiError("问答服务返回的数据格式不正确。") from error

    def check_connection(self) -> None:
        """检查服务文档端点是否可达，不触发模型或数据库调用。"""
        try:
            response = self._client.get(f"{self._base_url}/docs", timeout=5.0)
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise QueryApiError("问答服务连接超时。") from error
        except httpx.HTTPError as error:
            raise QueryApiError("问答服务不可达，请确认它正在 127.0.0.1:8000 运行。") from error

    def get_feature_flags(self) -> FeatureFlags:
        """读取持久化模块开关。"""
        try:
            response = self._client.get(f"{self._base_url}/runtime/features", timeout=self._timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            return FeatureFlags(
                faq_enabled=_required_bool(payload, "faq_enabled"),
                classifier_enabled=_required_bool(payload, "classifier_enabled"),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise QueryApiError("无法读取模块状态，请检查问答 API。") from error

    def update_feature_flags(self, flags: FeatureFlags) -> FeatureFlags:
        """保存模块开关，修改立即作用于后续问答。"""
        try:
            response = self._client.patch(
                f"{self._base_url}/runtime/features",
                json={
                    "faq_enabled": flags.faq_enabled,
                    "classifier_enabled": flags.classifier_enabled,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return FeatureFlags(
                faq_enabled=_required_bool(payload, "faq_enabled"),
                classifier_enabled=_required_bool(payload, "classifier_enabled"),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise QueryApiError("无法保存模块状态，请检查问答 API。") from error

    def close(self) -> None:
        """关闭由客户端自身创建的 HTTP 连接池。"""
        if self._owns_client:
            self._client.close()


def _parse_response(payload: Any) -> QueryResponse:
    if not isinstance(payload, dict):
        raise TypeError("response must be an object")
    citations_payload = payload.get("citations", [])
    if not isinstance(citations_payload, list):
        raise TypeError("citations must be a list")
    citations = tuple(_parse_citation(item) for item in citations_payload)
    web_payload = payload.get("web_citations", [])
    if not isinstance(web_payload, list):
        raise TypeError("web_citations must be a list")
    return QueryResponse(
        answer=_required_string(payload, "answer"),
        source=_required_string(payload, "source"),
        classification=_optional_string(payload.get("classification")),
        faq_confidence=float(payload["faq_confidence"]),
        classification_confidence=_optional_float(payload.get("classification_confidence")),
        fallback_reason=_optional_string(payload.get("fallback_reason")),
        citations=citations,
        web_citations=tuple(_parse_web_citation(item) for item in web_payload),
        web_search_used=bool(payload.get("web_search_used", False)),
    )


def _parse_citation(payload: Any) -> Citation:
    if not isinstance(payload, dict):
        raise TypeError("citation must be an object")
    return Citation(
        chunk_id=_required_string(payload, "chunk_id"),
        source=_required_string(payload, "source"),
        title_path=_required_string(payload, "title_path"),
        text=_required_string(payload, "text"),
    )


def _parse_web_citation(payload: Any) -> WebCitation:
    if not isinstance(payload, dict):
        raise TypeError("web citation must be an object")
    return WebCitation(
        title=_required_string(payload, "title"),
        url=_required_string(payload, "url"),
        snippet=_required_string(payload, "snippet"),
    )


def _required_string(payload: dict[str, Any], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError("optional value must be a string")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _required_bool(payload: Any, name: str) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get(name), bool):
        raise TypeError(f"{name} must be a boolean")
    return payload[name]


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"问答服务返回 HTTP {response.status_code}。"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, str) else f"问答服务返回 HTTP {response.status_code}。"
