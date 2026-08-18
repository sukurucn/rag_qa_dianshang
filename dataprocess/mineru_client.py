"""MinerU 精确解析 API 的签名上传客户端。"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

import httpx
from typing_extensions import Self


class MinerUError(RuntimeError):
    """MinerU 请求、解析或下载结果失败。"""


class MinerUClient:
    """通过 v4 批量上传 API 将本地 PDF/PPT 转为 Markdown。"""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://mineru.net/api/v4",
        poll_interval_seconds: float = 3.0,
        timeout_seconds: float = 600.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not token:
            raise ValueError("MINERU_API_KEY is required for PDF and PPT processing.")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._client = http_client or httpx.Client(timeout=60.0, follow_redirects=True)
        self._owns_client = http_client is None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        if self._owns_client:
            self._client.close()

    def parse_files(self, file_paths: list[Path]) -> list[str]:
        """上传一批本地文件、轮询任务并返回相应的 Markdown。"""
        if not file_paths:
            return []
        data_ids = [uuid.uuid4().hex for _ in file_paths]
        response = self._client.post(
            f"{self._base_url}/file-urls/batch",
            headers={**self._headers, "Content-Type": "application/json"},
            json={
                "files": [
                    {"name": path.name, "data_id": data_id}
                    for path, data_id in zip(file_paths, data_ids, strict=True)
                ],
                "model_version": "vlm",
                "enable_table": True,
                "enable_formula": True,
                "language": "ch",
            },
        )
        payload = _success_payload(response)
        batch_id = str(payload["batch_id"])
        upload_urls = payload.get("file_urls")
        if not isinstance(upload_urls, list) or len(upload_urls) != len(file_paths):
            raise MinerUError("MinerU returned an invalid signed upload URL list.")
        for path, upload_url in zip(file_paths, upload_urls, strict=True):
            with path.open("rb") as stream:
                upload_response = self._client.put(str(upload_url), content=stream.read())
            upload_response.raise_for_status()
        result_urls = self._wait_for_results(batch_id, data_ids)
        return [self._download_markdown(result_urls[data_id]) for data_id in data_ids]

    def _wait_for_results(self, batch_id: str, data_ids: list[str]) -> dict[str, str]:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            response = self._client.get(
                f"{self._base_url}/extract-results/batch/{batch_id}", headers=self._headers
            )
            payload = _success_payload(response)
            raw_results = payload.get("extract_result", [])
            results = raw_results if isinstance(raw_results, list) else [raw_results]
            completed: dict[str, str] = {}
            pending = False
            for result in results:
                if not isinstance(result, dict):
                    continue
                state = result.get("state")
                data_id = result.get("data_id")
                if state == "failed":
                    raise MinerUError(str(result.get("err_msg", "MinerU parsing failed.")))
                if state != "done":
                    pending = True
                    continue
                if isinstance(data_id, str) and isinstance(result.get("full_zip_url"), str):
                    completed[data_id] = result["full_zip_url"]
            if not pending and all(data_id in completed for data_id in data_ids):
                return completed
            time.sleep(self._poll_interval_seconds)
        raise MinerUError(f"MinerU timed out while parsing batch {batch_id}.")

    def _download_markdown(self, zip_url: str) -> str:
        response = self._client.get(zip_url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            markdown_name = next(
                (name for name in archive.namelist() if name.lower().endswith("full.md")), None
            )
            if markdown_name is None:
                raise MinerUError("MinerU result ZIP does not contain full.md.")
            return archive.read(markdown_name).decode("utf-8")


def _success_payload(response: httpx.Response) -> dict[str, object]:
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or body.get("code") != 0 or not isinstance(body.get("data"), dict):
        message = body.get("msg", "Unknown MinerU error") if isinstance(body, dict) else "Invalid response"
        raise MinerUError(str(message))
    return body["data"]
