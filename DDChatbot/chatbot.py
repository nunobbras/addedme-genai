from DDChatbot.logs import CustomLogger
from types import MethodType

custom_logger = CustomLogger("ChatbotLogger")

import os
import re
import uuid
from typing import List
import json
import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import cosine_similarity
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
import logging
import unidecode
import time
import numpy as np

from time import perf_counter
from DDChatbot import embeddings as emb
from DDChatbot.db.models import BaseDocument
from DDChatbot import embeddings as emb
from pathlib import Path
from DDChatbot.db import models
from mongoengine import connection
from datetime import datetime
from pytz import timezone


from DDChatbot.intentions import intentions_dict, intentions_funcs
from DDChatbot.chain_funcs import guardrails_default, scope_objective_default, correct_json_default

def tic():
    import time

    global global_tic, global_start, time_table
    global_tic = time.time()
    global_start = time.time()
    time_table = {}


def toc(event_name):
    global time_table, time_table, global_tic, global_timer
    curr = time.time()
    time_table[event_name] = curr - global_tic
    global_tic = curr
    print("event_name ", event_name, " - ", time_table[event_name])
    global_timer = curr - global_start



class ChatClass:
    def __init__(
        self,
        openai: openai,
        topics_dict: dict = {},
        chain_methods: list = [guardrails_default, scope_objective_default, correct_json_default],
        max_response_tokens: int = 1024,
        token_limit: int = 4096
    ) -> None:
        """
        Retrieve-then-read implementation, using the Cognitive Search and AzureOpenAI APIs directly. It first builds a
        new question based on the last question being made and the context of the conversation so far, then retrieves
        top documents from search, constructs a prompt with them, and then uses langchain with AzureOpenAI to generate
        a response with a 'map-reduce' strategy that builds a answerr sequentially over the top nr_docs documents found.
        """
        self.topics = [topics_dict[topic]["chat_topic_name"] for topic in topics_dict]
        self.openai = openai
        self.max_response_tokens = max_response_tokens
        self.token_limit = token_limit
        for m in chain_methods:
            setattr(self, m.__name__, MethodType(m, self))


    # Ensure a valid conversation_id

    def ensure_conversation_id(self, conversation_id):
        return (
            str(uuid.uuid4())
            if conversation_id is None or conversation_id == ""
            else conversation_id
        )


    @staticmethod
    def process_history(history: str) -> List[str]:
        """ """
        question = history[-1]["user"] if "user" in history[-1] else ""
        conversation = []
        if len(history) > 1:
            for interaction in history[:-1]:
                conversation.append({"role": "user", "content": interaction["user"]})
                conversation.append(
                    {
                        "role": "assistant",
                        "content": interaction["bot"].split("######")[0]
                        if "######" in interaction["bot"]
                        else interaction["bot"],
                    }
                )
        return question, conversation

    def run(
        self,
        payload: dict,
        conversation_id: str = None,
    ):
        global time_table
        time_table = {}
        mem_state = {}
        self.conversation_logger = custom_logger.get_logger()
        
        mem_state["payload"] = payload


        # Ensure a valid conversation_id
        # self.conversation_logger.info("Checking if conversation_id is valid")
        conversation_id = self.ensure_conversation_id(conversation_id)
        mem_state["conversation_id"] = conversation_id

        # Get best FAQs to answer a question
        tic()
        
        long_term_objective_OK, answer = self.guardrails_default(payload["long_term_objective"])

        toc("summarize_and_classify_question")

        if long_term_objective_OK != "True":
            mem_state["answer"] = answer
        else:
            
            answer = self.scope_objective_default(payload)
            try:
                answer_json = json.loads(answer)
            except Exception as e:
                print("exception loads: ", answer)
                answer_json = self.correct_json_default(answer)

            mem_state["answer"] = answer_json

        return self.process_response(mem_state)



    @staticmethod
    def process_response(
        mem_state: dict
    ) -> dict:
        """
        This method is a wrapper to validate the response provided in terms of
        format and produce and transform the answer to the correct format if
        needed
        """
        output_features = [ "payload", "answer", "conversation_id", "iteration_id" ]
        answer = {}
        for feat in output_features:
            if feat in mem_state:
                answer[feat] = mem_state[feat]
            else: 
                print("warning ", feat, "does not exist... ")
        return answer
