import json

def guardrails_default(
    self: object,
    question: str, 
) -> (str, str):
    conversation = []
    conversation.append(
        {
            "role": "system",
            "content": """
                Asssitant, Answer ONLY with Json format payload. 
                Your objective is to detect if a "long term objective is feasible by a human". If not, suggest an answer in the "answer" field.
                Your Answer should have this format: { "long_term_objective_OK": bool, "answer": string }
                Rules:
                1) If the long_term_objective field is not feasible by a human: { "long_term_objective_OK": "False", "answer": "This does not seem like a reasonable long term objective" } \n
                2) If the long_term_objective field is feasible by a human: { "long_term_objective_OK": "True", "answer": <nice comic and enthusiatsic comment about that objective<>> } \n
                "###\n\n"
            """
        }
    )

    conversation.append(
        {
            "role": "user",
            "content": (
                "Here is the user objective: \n \n#### \n \n"
                f"{question} \n \n"
            )
        }
    )
    
    response = self.openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=conversation,
        temperature=0,
        max_tokens= self.max_response_tokens,
    )

    try:
        resp_json = json.loads(response["choices"][0]["message"]["content"])
    except Exception as e:
        print("exception loads: ", response["choices"][0]["message"]["content"])
        self.conversation_logger.error(
            (
                "summarize_and_classify_incident error response",
                response["choices"][0]["message"]["content"],
            )
        )

    return resp_json.get("long_term_objective_OK"), resp_json.get("answer")