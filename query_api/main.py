"""用户问答 API 的本地启动入口。"""

from __future__ import annotations

import uvicorn

from query_api.app import create_app

app = create_app()


def main() -> None:
    """默认仅绑定 loopback，外部部署应通过认证网关转发。"""
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
