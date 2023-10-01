import json

def correct_json_default(
    self, payload: dict
) -> (str, str):
    conversation = []
    conversation.append(
        {
            "role": "system",
            "content": (
                """Assistant, you respond with corrected JSON strings. Do not include any explanations, Only provide a RFC8259 compliant JSON response""")

        })

    conversation.append(
        {
            "role": "user",
            "content": f"""{payload}"""
        })

    
    response = self.openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=conversation,
        temperature=0,
        max_tokens=self.max_response_tokens,
    )
    
    return response["choices"][0]["message"]["content"]