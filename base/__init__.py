"""应用基础设施的公共入口。"""

from base.config import Settings, settings
from base.logger import get_logger, logger

__all__ = ["Settings", "get_logger", "logger", "settings"]
