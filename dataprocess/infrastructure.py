"""文档处理前的 Docker Compose 基础设施预检。"""

from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_infrastructure(project_root: Path) -> None:
    """启动并等待 MySQL、Redis、Milvus 及其依赖健康。"""
    subprocess.run(
        ["docker", "compose", "up", "-d", "--wait"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
