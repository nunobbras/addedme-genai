import openai
import os
import logging

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from mongoengine import connect


topics_dict = {
    1: {
        "abrev": "FORUM",
        "doc_topic_name": "Fórum NOS",
        "chat_topic_name": "[1] Fórum NOS"
    }
}


def setup_openai():
    azure_credential = DefaultAzureCredential()
    AZURE_STORAGE_ACCOUNT = (
        os.environ.get("AZURE_STORAGE_ACCOUNT") or "mystorageaccount"
    )
    AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER") or "content"
    AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE") or "gptkb"
    AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "gptkbindex"
    AZURE_OPENAI_SERVICE = (
        os.environ.get("AZURE_OPENAI_SERVICE")
    )
    AZURE_OPENAI_GPT_DEPLOYMENT = (
        os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "text-davinci-003"
    )
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = (
        os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "gpt-35-turbo"
    )
    AZURE_BLOB_STORAGE_ACCOUNT = os.environ.get("AZURE_BLOB_STORAGE_ACCOUNT")
    OPENAI_API_KEY = (
        os.environ.get("OPENAI_API_KEY")
    )
    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION") or "2023-05-15"
    OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE") or "azure"

    # Used by the OpenAI SDK
    openai.api_type = OPENAI_API_TYPE
    openai.api_base = AZURE_OPENAI_SERVICE
    openai.api_version = OPENAI_API_VERSION
    openai.api_key = OPENAI_API_KEY

    return openai


def setup_openai_db():
    DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
    DB_NAME = os.environ.get("DB_NAME")
    # Connect to DB
    if DB_CONNECTION_STRING:
        logging.info(f"Connecting to {DB_CONNECTION_STRING}...")
        connect(DB_NAME, host=DB_CONNECTION_STRING)
    return setup_openai()
