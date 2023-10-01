"""Usage example of logger with database writes"""

import os
import random
from DDChatbot.logs import CustomLogger
from mongoengine import connect, disconnect
from dotenv import load_dotenv
from DDChatbot.db import models

HASH = random.getrandbits(8)
ID = f"my_example_logging_write_to_db_{HASH}"
load_dotenv()


def connect_to_db():
    connect(
        db=os.getenv("DB_NAME"),
        host=os.getenv("DB_CONNECTION_STRING")
    )


if __name__ == "__main__":

    custom_logger = CustomLogger("ChatbotLogger")
    logger = custom_logger.get_logger()
    logger.info("My Logger has been initialized")

    # Import only after setting up the logger!
    from DDChatbot.logs.example import auxiliary_script
    auxiliary_script.test_logger()

    logger.info("All logs updated")

    print(
        "Size of memory buffer:",
        len(custom_logger.get_memory_handler().buffer), "\n\n"
    )

    connect_to_db()

    custom_logger.save_memory_logs_to_mongo(ID)

    print(
        "\n\nObject saved to DB:",
        [
            document.to_mongo().to_dict()
            for document in models.ConversationLogs.objects().filter(_id=ID)
        ]
    )

    disconnect()
