"""应用统一日志配置。"""

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "log" / "app.log"
LOGGER_NAME = "ragAgentic"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logger() -> logging.Logger:
    """创建带控制台和文件处理器的 INFO 级应用日志器。"""
    app_logger = logging.getLogger(LOGGER_NAME)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    if app_logger.handlers:
        return app_logger

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.name = "console_handler"
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.name = "file_handler"
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)
    return app_logger


logger = configure_logger()


def get_logger(name: str | None = None) -> logging.Logger:
    """返回应用 logger 或其按模块命名的子 logger。"""
    return logger if not name else logger.getChild(name)
