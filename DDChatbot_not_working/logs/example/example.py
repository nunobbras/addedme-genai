"""Usage example of the logger"""

from datetime import datetime
from DDChatbot.logs import CustomLogger


if __name__ == "__main__":

    custom_logger = CustomLogger("ChatbotLogger")
    logger = custom_logger.get_logger()
    memory_handler = custom_logger.get_memory_handler()
    logger.info("My Logger has been initialized")

    # Import only after setting up the logger!
    from DDChatbot.logs.example import auxiliary_script
    auxiliary_script.test_logger()

    logger.info("All logs updated")

    print("Size of memory buffer:", len(memory_handler.buffer), "\n\n")

    for record in memory_handler.buffer:
        print({
            "logger_name": record.name,
            "level": record.levelname,
            "module": record.module,
            "line": record.lineno,
            "datetime": datetime.utcfromtimestamp(record.created),
            "message": record.getMessage()
        })
    print("\n\n")

    # Output the memory logs to the target
    # memory_handler.flush()
    custom_logger.flush()

    print("\n\nFinal size of memory buffer:", len(memory_handler.buffer))
    for record in memory_handler.buffer:
        print(record)
