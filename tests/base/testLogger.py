import logging

from base import get_logger, logger
from base.logger import LOG_PATH


def testExportsConfiguredInfoLogger() -> None:
    handler_names = {handler.name for handler in logger.handlers}

    assert logger.level == logging.INFO
    assert handler_names == {"console_handler", "file_handler"}
    assert get_logger("retrieval").name == "ragAgentic.retrieval"


def testWritesMessageToApplicationLog() -> None:
    message = "base logger test message"
    logger.info(message)

    for handler in logger.handlers:
        handler.flush()

    assert LOG_PATH.exists()
    assert message in LOG_PATH.read_text(encoding="utf-8")
