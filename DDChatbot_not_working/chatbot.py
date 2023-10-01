from DDChatbot.logs import CustomLogger

custom_logger = CustomLogger("ChatbotLogger")
conversation_logger = custom_logger.get_logger()

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


from DDChatbot.prompts import (
    similarity_prompt,
    main_prompt_improved_v1,
    guardrail_prompt,
)


def has_db_connection():
    try:
        connection.get_connection()
        return True
    except:
        return False


def tic():
    import time

    global global_tic, global_start, time_table
    global_tic = time.time()
    global_start = time.time()
    time_table = {}


    global time_table, time_table, global_tic, global_timer
    curr = time.time()
    time_table[event_name] = curr - global_tic
    global_tic = curr
    global_timer = curr - global_start


class EmbeddingsRetriever(BaseRetriever):
    def __init__(self, openai: openai, topics_dict: dict):
        self.topics_dict = topics_dict
        self.FAQS = None
        self.openai = openai

    def get_relevant_documents(self, query: str) -> List[Document]:
        pass

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        pass

    def search_query(
        self, df: pd.DataFrame, query: str, n: int = 10, threshold: float = 0.8
    ):
        query_norm = unidecode.unidecode(query).lower()

        query_vector = get_embedding(query_norm)["data"][0]["embedding"]

        df["similarities_q"] = df.embeddings.apply(
            lambda x: cosine_similarity(x, query_vector)
        )
        df["similarities_qa"] = df.embeddings_qa.apply(
            lambda x: cosine_similarity(x, query_vector)
        )

        df["similarities"] = df[["similarities_qa", "similarities_q"]].max(axis=1)

        data = df.sort_values("similarities", ascending=False)

        res = data.head(n).loc[data["similarities"] >= threshold, :]

        sec_res = res.iloc[1:4, :]
        return res, sec_res

    def get_faqs_and_followups(self, question: str, n: int, topic_number: int = None):
        # [TODO] should use history instead of question
        if self.FAQS is not None:
            # Topic Filtering
            # self.sub_faqs = (
            #     self.FAQS.loc[self.FAQS["topic_number"] == topic_number, :].copy()
            #     if topic_number
            #     else self.FAQS.copy()
            # )
            resulting_faqs, resulting_folllowups = self.search_query(
                self.FAQS, question, n=n, threshold=0.60
            )
            resulting_faqs["page_content"] = resulting_faqs["all"]

            followups = []
            for _, touch in resulting_folllowups.iterrows():
                title = touch["title"]
                title = re.sub(r"part [0-9]+ of [0-9]+", "", title)
                followups.append(title)
            followups = list(set(followups))
            docfaqs = []
            for _, faq in resulting_faqs.iterrows():
                docfaqs.append(
                    Document(
                        page_content=faq["page_content"],
                        metadata={
                            "title": faq["id"] + " " + faq["title"],
                            "score": faq["similarities"],
                            "source": faq["source"],
                            "id": faq["id"],
                        },
                    )
                )
        else:
            raise Exception("No FAQS - run get_all_faqs_w_embeddings")
        return (docfaqs, followups)

    def get_relevant_texts(self, question, topic_number):
        faqs, followups = self.get_faqs_and_followups(question, 20, topic_number)
        return faqs

    def get_all_faqs_w_embeddings(
        self,
        force_rebuild_embs=False,
        embs_filename=None,
        update_embs=False,
        faqs_file_path=None,
        save_to_db=False,
    ):
        if embs_filename is None:
            path = Path(__file__).parent
            path.mkdir(parents=True, exist_ok=True)
            embs_filename = path.joinpath("../data/QnA_NOS_embeddings.csv")

        if force_rebuild_embs:
            print("rebuilding embs...")
            if faqs_file_path is None:
                faqs_file_path = "ALL_FAQS.xlsx"
            pd_docs = pd.read_excel(faqs_file_path)
            pd_docs = emb.process_base_df(pd_docs)
            FAQS = emb.build_embedings_w_topics(pd_docs, self.topics_dict, openai)
            self.save_faqs(FAQS, embs_filename, save_to_db=save_to_db)
            self.FAQS = FAQS
        elif update_embs:
            FAQS = self.load_faqs(embs_filename)
            logging.info("updating embs...")
            if faqs_file_path is None:
                faqs_file_path = "ALL_FAQS.xlsx"
            pd_docs = pd.read_excel(faqs_file_path)
            pd_docs = emb.process_base_df(pd_docs[pd_docs["update"] == 1])
            print(f"updating embeddings for documents: {pd_docs['id'].to_list()}")
            new_FAQS = emb.build_embedings_w_topics(pd_docs, self.topics_dict, openai)
            FAQS = pd.concat([FAQS[~FAQS["id"].isin(new_FAQS.id)], new_FAQS])
            self.save_faqs(FAQS, embs_filename, save_to_db=save_to_db)
            self.FAQS = FAQS
        else:
            if self.FAQS is None:
                FAQS = self.load_faqs(embs_filename)
                self.FAQS = FAQS

    def save_faqs(
        self,
        FAQS=None,
        embs_file_path="../data/QnA_NOS_embeddings.csv",
        save_to_db=False,
    ):
        if FAQS is None:
            FAQS = self.FAQS

        # TODO: Find better strategy to prevent conversion to int problems
        FAQS.loc[FAQS.topic_number.isna(), "topic_number"] = 999_999
        FAQS.loc[FAQS.topic.isna(), "topic"] = "no topic"
        FAQS.loc[FAQS.source.isna(), "source"] = "no source"
        FAQS.loc[FAQS.category_id.isna(), "category_id"] = -1
        FAQS.loc[FAQS.sub_category_id.isna(), "sub_category_id"] = -1
        FAQS.loc[FAQS.category_name.isna(), "category_name"] = "no category"
        FAQS.loc[FAQS.sub_category_name.isna(), "sub_category_name"] = "no sub category"

        if has_db_connection() and save_to_db:
            print("updating db...")
            for _, row in FAQS.iterrows():
                # create a new BaseDocument
                BaseDocument.objects(id=str(row["id"])).update_one(
                    set__title=row["title"],
                    set__content=row["content"],
                    set__type=row["type"],
                    set__all=row["all"],
                    set__source=row["source"],
                    set__topic=row["topic"],
                    set__title_embs=row["title_embs"],
                    set__embeddings=row["embeddings"],
                    set__all_embs=row["all_embs"],
                    set__embeddings_qa=row["embeddings_qa"],
                    set__topic_number=int(row["topic_number"])
                    if row["topic_number"] and row["topic_number"] != np.nan
                    else -1,
                    set__category_id=int(row["category_id"])
                    if row["category_id"] and row["category_id"] != np.nan
                    else -1,
                    set__sub_category_id=int(row["sub_category_id"])
                    if row["sub_category_id"] and row["sub_category_id"] != np.nan
                    else -1,
                    set__category_name=row["category_name"],
                    set__sub_category_name=row["sub_category_name"],
                    upsert=True,
                )
        else:
            FAQS.to_csv(embs_file_path, index=False)

    def load_faqs(self, embs_filename=None):
        if has_db_connection():
            FAQS = pd.DataFrame(
                [doc.to_mongo().to_dict() for doc in models.BaseDocument.objects()]
            )
            FAQS["id"] = FAQS["_id"]
            logging.info(f"loaded {len(FAQS)} documents from db")
        else:
            if embs_filename is None:
                path = Path(__file__).parent
                path.mkdir(parents=True, exist_ok=True)
                embs_filename = path.joinpath("data/embs.csv")
            FAQS = pd.read_csv(embs_filename)
            FAQS.loc[:, "embeddings"] = FAQS.loc[:, "embeddings"].apply(json.loads)
            FAQS.loc[:, "embeddings_qa"] = FAQS.loc[:, "embeddings_qa"].apply(
                json.loads
            )
            logging.info(f"loaded {len(FAQS)} documents from file")
        return FAQS


class ChatReadRetrieveReadApproach:
    def __init__(
        self,
        openai: openai,
        topics_dict: dict,
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
        self.embRetriever = EmbeddingsRetriever(openai=openai, topics_dict=topics_dict)
        self.openai = openai
        self.max_response_tokens = max_response_tokens
        self.token_limit = token_limit

    def rate(
        self, conversation_id: str, iteration_id: str, like: bool, reason: str
    ) -> None:
        logging.info(f"context: {conversation_id}, {iteration_id}, like: {like}")

        if has_db_connection():
            chat = models.ChatLike(
                conversation_id=conversation_id,
                iteration_id=iteration_id,
                like=like,
                reason=reason,  # TODO add a reason to frontend
            )
            chat.save()

    def stream_prep(self, data):
        return json.dumps({"data": data}) + "\n\n"

    def stream_prep_system(self, data):
        return json.dumps({"system": json.dumps(data)}) + "\n\n"

    # def yield_sentence(self, text):
    #     for t in text.split(" "):
    #         yield self.stream_prep(t)

    def stream(
        self,
        history: List[dict],
        max_words: int = None,
        conversation_id: str = None,
        interaction_type: str = None,
        is_avatar: bool = None,
        is_avatar_sound: bool = None,
        layout_width: str = None,
        layout: str = None,
    ):
        start_time = perf_counter()
        global time_table
        time_table = {}
        thought_process = {}
        docs = []
        followups = []


        question, conversation = self.process_history(history)
        logging.info(f"Original Question: {question}, conversation_id: {conversation_id}")
        conversation_logger.info(f"Original input: {question}")

        thought_process["raw_question"] = question
        thought_process["previous_conversation"] = conversation

        # Check validity of the question
        conversation_logger.info("Checking if input is valid")
        is_valid, response = self.check_valid_question(question, conversation_id)
        if not is_valid:
            return response

        # Ensure a valid conversation_id
        conversation_logger.info("Checking if conversation_id is valid")
        conversation_id = self.ensure_conversation_id(conversation_id)
        thought_process["conversation_id"] = conversation_id
        yield self.stream_prep_system({"conversation_id": str(conversation_id)})

        # Get best FAQs to answer a question
        tic()

        conversation_logger.info("Summarizing and classyfing input")
        summarized_question, intention = self.summarize_and_classify_question(
            question, conversation
        )
        logging.info(("0) Summarized question:", summarized_question, intention))
        thought_process["summarized_question"] = summarized_question
        thought_process["initial_intention"] = intention

        if intention != "LEGIT":
            conversation_logger.info(
                "Applying guardrail prompt (INTENTION != LEGIT)"
            )
            ga = self.basic_inline_guardail_response(
                summarized_question, intention, True
            )
            expl = ""
            

        else:
            yield self.stream_prep_system({"wait": "summarized"})
            n = 5
            conversation_logger.info(f"Fetching Top {n} FAQs (1st attempt)")
            docs, followups = self.fetch_faqs(summarized_question, n)
            logging.info(("0) FAQS ", len(docs)))

            conversation_logger.info(
                "Starting similarity validation (1st attempt)"
            )
            similarity, expl, _ = self.match_similarity(
                summarized_question, docs
            )

            logging.info(
                (
                    "1) FIRST SIMILARITY MATCH: similarity ",
                    similarity,
                    " new_request ",
                    _,
                    " len ",
                    len(docs),
                )
            )
            if len(docs) != 0:
                thought_process["small_n_faqs_similarity_expl"] = ""
                for i, (doc, cl) in enumerate(
                    [(docs[q["n"] - 1], q["cl"]) for q in expl]
                ):
                    docs_score = f'Doc {i}: {doc.metadata["score"]} - {doc.metadata["title"]} - {cl}'
                    logging.info(
                        docs_score
                    )
                    conversation_logger.info(
                        f'FAQ {doc.metadata["id"]}: '
                        f'{doc.metadata["score"]} - '
                        f'{doc.metadata["title"]}'
                    )
                    thought_process["small_n_faqs_similarity_expl"] += docs_score + "\n"
            else:
                conversation_logger.info(
                        "No relevant FAQs found (1st attempt)"
                    )


            if not similarity:
                yield self.stream_prep("Por favor espere mais uns segundos ...\n")
                n = 10
                conversation_logger.info(f"Fetching Top {n} FAQs (2nd attempt)")
                docs, followups = self.fetch_faqs(summarized_question, n)
   
                conversation_logger.info(
                    "Starting similarity validation (2nd attempt)"
                )
                similarity, expl, new_request = self.match_similarity(
                    summarized_question, docs
                )

                logging.info("3) SECOND SIMILARITY MATCH:")
                thought_process["large_n_faqs_similarity"] = "True"
                if len(docs) != 0:
                    thought_process["large_n_faqs_similarity_expl"] = ""
                    for i, (doc, cl) in enumerate(
                        [(docs[q["n"] - 1], q["cl"]) for q in expl]
                    ):
                        docs_score = f'Doc {i}: {doc.metadata["score"]} - {doc.metadata["title"]} - {cl}'
                        logging.info(
                            docs_score
                        )
                        conversation_logger.info(
                            f'FAQ {doc.metadata["id"]}: '
                            f'{doc.metadata["score"]} - '
                            f'{doc.metadata["title"]}'
                        )
                        thought_process["large_n_faqs_similarity_expl"] += docs_score + "\n"
                else:
                    conversation_logger.info(
                        "No relevant FAQs found (2nd attempt)"
                    )

            else:
                thought_process["small_n_faqs_similarity"] = "True"
                logging.info("3) FIRST ATTEMPT OK")

            # Generate answer
            if similarity: 
                conversation_logger.info("Generating answer")
                ga = self.generate_response(
                    question,
                    conversation,
                    [docs[q["n"] - 1] for q in expl if q["cl"] == "SIMILAR"][
                        :5
                    ],  # TODO reduce FAQS only if needed for the tokens
                    max_words,
                    streaming=True,
                )

            else: 
                conversation_logger.info(f"Asking for clarification")
                ga = new_request
                thought_process["final_clarification_request"] = new_request

        complete_text = ""
        if isinstance(ga, str):
            print("ga str", ga)
            conversation_logger.info(f"Chatbot Response: {ga}")
            yield self.stream_prep(ga)
        else:
            print("ga arr", ga)
            final_response_complete = ""
            for line in ga:
                chunk = line["choices"][0].get("delta", {}).get("content", "")
                if chunk:
                    final_response_complete += chunk
                    yield self.stream_prep(chunk)

            conversation_logger.info(
                f"Chatbot Response: {final_response_complete}"
            )

        yield self.stream_prep_system(
            {"thought_process": thought_process}
        )

        yield self.stream_prep_system(
            {"data_points": [doc.page_content for doc in docs]}
        )   
        yield self.stream_prep_system(
            {
                "faqs": [
                    "{score: .1f} - ".format(score=(doc.metadata["score"] * 100))
                    + doc.metadata["title"]
                    for doc in docs
                ]
            }
        )


        conversation_logger.info(
            f"Total time: "
            f"{(perf_counter()-start_time):0.2f} seconds."
        )
        conversation_logger.info("Saving user interactions to Mongo Database")

        self.save_conversation_to_db(
            conversation,
            question,
            summarized_question,
            complete_text,
            interaction_type,
            is_avatar,
            is_avatar_sound,
            layout_width,
            layout,
            conversation_id,
            expl,
            docs
        )


        logging.info(f"execution time table: {time_table}, global: {global_timer}")

        yield self.stream_prep_system({"followups": followups})

        if has_db_connection():
            conversation_logger.info("Saving conversation logs to Mongo Database")
            now = datetime.now(timezone("Europe/Lisbon")).strftime("%Y%m%d_%H%M%S")
            custom_logger.save_memory_logs_to_mongo(
                id=f"{now}-{conversation_id}", conversation_id=conversation_id
            )
        custom_logger.flush()  # Send outputs to terminal and clean buffer

    def run(
        self,
        history: List[dict],
        max_words: int = None,
        conversation_id: str = None,
        interaction_type: str = None,
        is_avatar: bool = None,
        is_avatar_sound: bool = None,
        layout_width: str = None,
        layout: str = None,
    ) -> dict:
        global time_table
        time_table = {}

        docs = []
        followups = []

        question, conversation = self.process_history(history)
        logging.info(f"Original Question: {question}")

        # Check validity of the question
        is_valid, response = self.check_valid_question(question, conversation_id)
        if not is_valid:
            return response

        # Ensure a valid conversation_id
        conversation_id = self.ensure_conversation_id(conversation_id)

        # Get best FAQs to answer a question
        tic()
        summarized_question = ""
        summarized_question, intention = self.summarize_and_classify_question(
            question, conversation
        )

        logging.info(("0) Summarized question:", summarized_question, intention))

        if intention != "LEGIT":
            ga = self.basic_inline_guardail_response(
                summarized_question, intention, False
            )
            expl = ""
            similarity = []

        else:
            docs, followups = self.fetch_faqs(summarized_question, 5)
            logging.info(("0) FAQS ", len(docs)))

            similarity, expl, new_request = self.match_similarity(
                summarized_question, docs
            )

            if len(docs) != 0:
                for i, (doc, cl) in enumerate(
                    [(docs[q["n"] - 1], q["cl"]) for q in expl]
                ):
                    logging.info(
                        f'Doc {i}: {doc.metadata["score"]} - {doc.metadata["title"]} - {cl}'
                    )

            if not similarity:
                docs, followups = self.fetch_faqs(summarized_question, 10)
                print("docs ", len(docs))
                similarity, expl, new_request = self.match_similarity(
                    summarized_question, docs
                )

                logging.info("3) SECOND SIMILARITY MATCH:")
                if len(docs) != 0:
                    for i, (doc, cl) in enumerate(
                        [(docs[q["n"] - 1], q["cl"]) for q in expl]
                    ):
                        logging.info(
                            f'Doc {i}: {doc.metadata["score"]} - {doc.metadata["title"]} - {cl}'
                        )

            else:
                logging.info("3) FIRST ATTEMPT OK")

            # Generate answer
            if similarity:
                ga = self.generate_response(
                    question,
                    conversation,
                    [docs[q["n"] - 1] for q in expl if q["cl"] == "SIMILAR"][
                        :5
                    ],  # TODO reduce FAQS only if needed for the tokens
                    max_words,
                    streaming=True,
                )
            else:
                ga = new_request

            logging.info(
                f"Summarized question {summarized_question} - similarity: {similarity}, expln: {expl} /////"
            )

        sources = self.extract_sources()

        response = ga["choices"][0]["message"]["content"]

        print("resp ", response)

        response += (
            " ###### "
            + sources
            + " "
            + "".join(f"<<{fu}>>" for fu in followups)
        )

        chat = self.save_conversation_to_db(
            conversation,
            question,
            summarized_question,
            response,
            interaction_type,
            is_avatar,
            is_avatar_sound,
            layout_width,
            layout,
            conversation_id,
            expl,
            docs
        )


        logging.info(f"execution time table: {time_table}")
        iteration_id = chat.pk if chat is not None else ""

        return self.process_response(
            question=question,
            summarized_question=summarized_question,
            response=response,
            docs=docs,
            searched=sources,
            topics=[],
            followups=followups,
            similarity=similarity if similarity else [],
            expl=expl,
            conversation_id=conversation_id,
            iteration_id=str(iteration_id),
        )

    def stats(
        self,
        conversation_id: str,
        iteration_id: str,
        backend: float = None,
        avatar_tts: float = None,
        avatar: float = None,
    ) -> None:
        logging.info(f"context: {conversation_id}, {iteration_id}")

        if has_db_connection():
            chat = models.ChatStats(
                conversation_id=conversation_id,
                iteration_id=iteration_id,
                backend=backend,
                avatar_tts=avatar_tts,
                avatar=avatar,
            )
            chat.save()

    # Ensure a valid conversation_id

    def ensure_conversation_id(self, conversation_id):
        return (
            str(uuid.uuid4())
            if conversation_id is None or conversation_id == ""
            else conversation_id
        )

    # Check if the question is valid

    def check_valid_question(self, question, conversation_id):
        if question == "":
            return False, self.process_response(
                question="-",
                summarized_question="-",
                response="Desculpe nao percebi a pergunta. Por favor repita.",
                conversation_id=conversation_id,
            )
        return True, None

    # Fetch FAQs based on the question

    def fetch_faqs(self, question, n):
        self.embRetriever.get_all_faqs_w_embeddings()
        return self.embRetriever.get_faqs_and_followups(question, n)


    def extract_sources(self, docs):
        sources = set()
        if len(docs) > 0 and docs is not None:
            for i, doc in enumerate(docs):
                logging.info(
                    f'Doc {i}: {doc.metadata["score"]} - {doc.metadata["title"]}'
                )
                sources.add(doc.metadata["source"])
        return sources

    # Extracted from the run() method to save to DB
    def save_conversation_to_db(
        self,
        conversation,
        question,
        summarized_question,
        response,
        interaction_type,
        is_avatar,
        is_avatar_sound,
        layout_width,
        layout,
        conversation_id,
        expl,
        docs
    ):
        if has_db_connection():
            chat = models.ChatEntry(
                history=conversation,
                question=question,
                conversation_id=conversation_id,
                response=response,
                summarized_question=summarized_question,
                base_documents=[
                    models.DocumentWithSimilarity(
                        similarity=q["cl"] == "SIMILAR",
                        document=docs[q["n"] - 1].metadata["id"],
                    )
                    for q in expl
                    if len(docs) > 0 and docs is not None
                ],
                interaction_type=interaction_type,
                is_avatar=is_avatar,
                is_avatar_sound=is_avatar_sound,
                layout_width=layout_width,
                layout=layout,
                time_table=time_table,
            )
            chat.save()
            return chat

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

    def summarize_and_classify_question(
        self, question: str, history: list
    ) -> (str, str):
        conversation = []
        conversation.append(
            {
                "role": "system",
                "content": (
                    """Assistant, you analyse the client intent considering
                    all conversation and resume it into a new made as the client, 
                    that expresses the client intent. You anwer in 'Portuguese from
                    Portugal' and ONLY answer in JSON. client intentions should be only one of these:
                    - [SALUTE] - Intent to salute
                    - [GOODBYE] - Intent to say good bye or not to talk
                    - [OFFEND] - Intent to offend with explicit wording
                    - [ASSISTANT] Intent to know more about the you, the assistant
                    - [LEGIT] - Intent to talk about telecomunications, equipment, the company NOS and its services and products

                    """
                ),
            }
        )

        conversation.append(
            {
                "role": "user",
                "content": (
                    "Here is the conversation (the Last is more Recent!): \n \n#### \n \n"
                    f"{history[-5:]} \n \n"
                    "Heres the last interaction: \n \n#### \n \n"
                    f"{question}"
                    "\n \n#### \n \n"
                    "Please return a reformulated question in the field 'question'. Add the client intention  in the field 'intention' like this example:"
                    """{{
                        "question": <<reformulated question>>,
                        "intention": "SALUTE"
                        }}
                    """
                ),
            }
        )
        print("conversation", conversation)
        response = self.openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation,
            temperature=0,
            max_tokens=self.max_response_tokens,
        )

        try:
            resp_json = json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            conversation_logger.error(
                (
                    "summarize_and_classify_question error response",
                    response["choices"][0]["message"]["content"],
                )
            )

        conversation_logger.info(
            f"Summarized input: {resp_json.get('question')}"
        )
        conversation_logger.info(
            f"Detected intention: {resp_json.get('intention')}"
        )

        return resp_json.get("question"), resp_json.get("intention")

    def get_topic_number(self, query_history: list) -> int:
        query_history_usable = "\n".join(query_history[-3:])
        conversation = []
        conversation.append(
            {
                "role": "system",
                "content": (
                    "Assistant you are a a classifier programmed to classifiy a prompt according with one of the following topics: \n"
                    f"{self.topics} \n \n"
                ),
            }
        )

        conversation.append(
            {
                "role": "user",
                "content": (
                    f"Select the topic for the last question here: {query_history_usable}. \n"
                    "Return ONLY the number of the topic inside brackets like this: [1] "
                ),
            }
        )
        # print("conversation: ", conversation)

        response = self.openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation,
            temperature=0,
            max_tokens=self.max_response_tokens,
        )

        resp = response["choices"][0]["message"]["content"]
        return int(resp.split("[")[1].split("]")[0])

    def basic_inline_guardail_response(
        self,
        question: str,
        intention: str,
        streaming: bool,
    ):
        """ """
        conversation = []

        conversation.append(
            {
                "role": "system",
                "content": guardrail_prompt(),
            }
        )

        conversation.append(
            {
                "role": "user",
                "content": (f"{question}, with intention {intention}"),
            }
        )

        response = self.openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation,
            temperature=0,
            max_tokens=self.max_response_tokens,
            stream=streaming,
        )

        conversation_logger.info(
            f"Guardrail prompt returned response: {response}"
        )

        return response

    def generate_response(
        self,
        question: str,
        history: List[dict],
        docs: List[Document],
        max_words: int,
        streaming: bool,
    ):
        """ """
        conversation = []

        facts = (
            "\n".join(
                f"{i + 1}. ```{doc.page_content}````" for i, doc in enumerate(docs)
            )
            if len(docs) > 0
            else "1. No information"
        )

        conversation.append(
            {
                "role": "system",
                "content": main_prompt_improved_v1(facts, len(docs)),
            }
        )

        conversation.append(
            {
                "role": "user",
                "content": (f"{question}"),
            }
        )
        # conversation = self.conversation_tokens_reducer(conversation)
        # print("conversation - \n", conversation)

        response = self.openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation,
            temperature=0,
            max_tokens=self.max_response_tokens,
            stream=streaming,
        )

        return response

    def match_similarity(self, question: str, docs: List[dict]):
        """ """

        max_score = max([d.metadata["score"] for d in docs])
        if max_score > 0.8:
            logging.info("match_similarity OK - Scores over threshold")
            conversation_logger.info(
                f"Max score ({max_score}) > 0.8 (Similarity OK)"
            )
            return (
                True,
                [
                    {"n": i, "cl": "SIMILAR"}
                    for i, d in enumerate(docs)
                    if d.metadata["score"] > 0.8
                ],
                "",
            )

        logging.info("match_similarity - testing LLM similarity...")
        conversation_logger.info(
            f"Max score ({max_score}) <= 0.8 (Similarity NOK)"
        )

        conversation = []

        conversation_logger.info("Classifying FAQs as SIM/NOT-SIM")
        questions = (
            "\n".join(
                f"TOPIC {i + 1} - {doc.page_content.split('Answer:')[0]}"
                for i, doc in enumerate(docs)
            )
            if len(docs) > 0
            else "1. No information"
        )
        conversation.append(
            {
                "role": "user",
                "content": similarity_prompt(question, questions, len(docs)),
            }
        )

        print("conversation", conversation)

        response = self.openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation,
            temperature=0,
            max_tokens=self.max_response_tokens,
        )

        try:
            resp_json = json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            conversation_logger.error(
                (
                    "match_similarity error response",
                    response["choices"][0]["message"]["content"],
                )
            )

        return (
            True
            if sum([a["cl"] == "SIMILAR" for a in resp_json.get("INTENT", [])]) > 0
            else False,
            resp_json.get("INTENT", []),
            resp_json.get("REQ", ""),
        )

    @staticmethod
    def process_response(
        question: str,
        summarized_question: str,
        response: str,
        docs: str = [],
        searched: str = None,
        topics: List[str] = [],
        followups: List[str] = [],
        similarity: List[str] = [],
        expl: List[str] = [],
        conversation_id: str = None,
        iteration_id: str = None,
    ) -> dict:
        """
        This method is a wrapper to validate the response provided in terms of
        format and produce and transform the answer to the correct format if
        needed
        """
        process = ""
        if searched is not None:
            process = (
                "Encontrei as seguintes palavras chave baseadas na pergunta feita. \n"
                f"Palavras chave: {searched} \n"
                "De seguida pesquisei por essa palavras nas fontes disponíveis "
                "e atualizadas sobre informação de serviços portugueses. \n"
            )
            if len(topics) > 0:
                process += (
                    "A pesquisa foi restringida aos seguintes tópicos: "
                    ", ".join(f"{topic}" for topic in topics)
                )
            process += "\n Com base na informação adquirida gerei a resposta final."
        answer = {
            "data_points": [doc.page_content for doc in docs],
            "question": question,
            "summarized_question": summarized_question,
            "answer": response,
            "thoughts": process,
            "topics": topics,
            "similarity": similarity,
            "expl": expl,
            "faqs": [
                "{score: .1f} - ".format(score=(doc.metadata["score"] * 100))
                + doc.metadata["title"]
                for doc in docs
            ],
            "follow-ups": followups,
            "conversation_id": conversation_id,
            "iteration_id": iteration_id,
        }
        return answer

    def conversation_tokens_reducer(self, conversation):
        conv_history_tokens = self.num_tokens_from_messages(conversation)
        while conv_history_tokens + self.max_response_tokens >= self.token_limit:
            try:
                del conversation[2]
            except IndexError:
                logging.warning("One of the prompts is to big")
            try:
                del conversation[2]
            except IndexError:
                logging.warning("One of the prompts is to big")
            conv_history_tokens = self.num_tokens_from_messages(conversation)

        return conversation

    @staticmethod
    def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
