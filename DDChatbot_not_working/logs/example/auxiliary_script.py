"""Auxiliary script that uses the logger in the example.py"""

import logging
hostlogger = logging.getLogger("ChatbotLogger.module")


def test_logger():
    hostlogger.info("Logger has been initialised in the host module")
    hostlogger.warning("Logger has been initialised in the host module")
    hostlogger.error("Logger has been initialised in the host module")
