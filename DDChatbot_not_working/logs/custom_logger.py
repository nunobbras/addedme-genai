import logging
from logging.handlers import MemoryHandler
from typing import Optional
from datetime import datetime
from DDChatbot.db import models

LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CAPACITY = 250
FLUSH_LEVEL = logging.CRITICAL


class CustomLogger():
    """Custom logger used to save conversation interactions"""
    def __init__(self, logger_name) -> None:
        self.logger, self.memory_handler = self._setup_logger(logger_name)

    def save_memory_logs_to_mongo(
        self, id: str, conversation_id: Optional[str] = None
    ):
        """Save logs in memory buffer to mongo Database"""
        models.ConversationLogs(
            _id=id,
            conversation_id=conversation_id,
            logs=[
                models.LogInfo(
                    logger_name=record.name,
                    level=record.levelname,
                    module=record.module,
                    line=record.lineno,
                    timestamp=datetime.utcfromtimestamp(record.created),
                    message=record.getMessage()
                )
                for record in self.memory_handler.buffer
            ]
        ).save()

    def flush(self):
        """Send output to stream handler and clear memory buffer"""
        self.memory_handler.flush()

    def get_logger(self):
        """Returns the logger"""
        return self.logger

    def get_memory_handler(self):
        """Returns the memory_handler"""
        return self.memory_handler

    def _setup_logger(self, logger_name: Optional[str] = None):
        """Setup logger"""
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        stream_handler = self._setup_stream_handler()
        memory_handler = self._setup_memory_handler(stream_handler)

        logger.addHandler(memory_handler)

        return logger, memory_handler

    @staticmethod
    def _setup_memory_handler(
        stream_handler: Optional[logging.StreamHandler] = None
    ) -> MemoryHandler:
        """Setup memory handler to save logs in a memory buffer"""
        memory_handler = MemoryHandler(
            capacity=CAPACITY,
            flushLevel=FLUSH_LEVEL,
            target=stream_handler
        )
        memory_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))

        return memory_handler

    @staticmethod
    def _setup_stream_handler() -> logging.StreamHandler:
        """Setup stream handler to print the logs in the terminal"""
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))

        return stream_handler
