"""uvicorn 本地启动入口。"""

from __future__ import annotations

import uvicorn

from admin_api.app import create_app

app = create_app()


def main() -> None:
    """只绑定 loopback，避免未经认证的管理接口暴露到局域网。"""
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
