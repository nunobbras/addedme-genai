import os
import mimetypes
import time
import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from DDChatbot.config import setup_openai_db, topics_dict
from DDChatbot.chatbot import ChatClass
from dotenv import find_dotenv, load_dotenv

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
load_dotenv()

openai = setup_openai_db()
chatbot = ChatClass(openai, topics_dict=topics_dict)

cors = CORS(
    app,
    resources={
        r"*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "*"
            ]
        }
    },
)

load_dotenv(find_dotenv())

@app.route("/", methods=["GET"])
def live_beat():
    return "here API genAI!!"


@app.route("/chat", methods=["POST"])
def chat():
    approach = "embrrr"
    try:
        print("conversation_id", request.json)
        r = chatbot.run(
            request.json.get("history"),
            # request.json.get("overrides"),
            request.json.get("max_words", None),
            request.json.get("conversation_id", None),
            request.json.get("interaction_type", None),
            request.json.get("is_avatar", None),
            request.json.get("is_avatar_sound", None),
            request.json.get("layout_width", None),
            request.json.get("layout", None),
        )
        print("docs: ", [doc.metadata["title"] for doc in chatbot.docs])
        response = jsonify(r)
    except Exception as e:
        logging.exception("Exception in /chat")
        response = jsonify({"error": str(e)}), 500

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
