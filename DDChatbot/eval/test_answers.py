import json
import pandas as pd

from DDChatbot.prompts import generate_eval_prompt


def get_QA_performance(chat, data, prompt_generator: callable = generate_eval_prompt):
    conversation = []

    conversation.append(
        {
            "role": "system",
            "content": prompt_generator(data),
        }
    )
    try:
        response = chat.openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=conversation,
            temperature=0,
            max_tokens=chat.max_response_tokens,
        )
        response = response["choices"][0]["message"]["content"]

    except Exception as e:
        print("error - ", data)
        print("error type - ", e)
        response = '{"class": 0, "explanation": "error"}'

    return response


def eval_chatbot_extra_data(chat, assessment_data, extra_data):
    performance_raw_data = []
    results = []
    for qa, exdata in zip(assessment_data, extra_data):
        print(qa["question"])
        performance_raw_data.append(get_QA_performance(chat, qa))
        res = qa | json.loads(performance_raw_data[-1]) | exdata
        results.append(res)
    return results


def eval_chatbot(chat, assessment_data, prompt_generator=generate_eval_prompt):
    performance_raw_data = []
    results = []
    errors = []
    for qa in assessment_data:
        print(qa["question"])
        perf = get_QA_performance(chat, qa, prompt_generator)
        try:
            res = qa | json.loads(perf)
            performance_raw_data.append(perf)
            results.append(res)
        except Exception as e:
            errors.append({"error": e, "qa": qa, "eval": perf})
    return results, performance_raw_data, errors


def print_resp(r):
    print("----")
    print("question ", r["question"])
    print("----")
    print("anwser ", r["answer"], "\n")
    print("similarity ", r["similarity"], "\n")
    for r_ in r["faqs"]:
        print("faq ", r_)


def run_all_QA(chat, QA_dict, history=[]):
    answers = []
    for index, qa in enumerate(QA_dict):
        this_history = history.copy()
        this_history.append(
            {
                "bot": None,
                "user": qa["Q"],
            }
        )

        print(index, this_history)
        try:
            r = chat.run(this_history)
            answers.append(r)
        except Exception as e:
            print(e)
    return answers


def show_answers(answers):
    for index, ans in enumerate(answers):
        # print(questions_and_answers.QA_main[index], "\n", ans)
        print_resp(ans)
        print("--------------------")


def prepare_answers_df(answers):
    answersdf = pd.DataFrame(answers)
    answersdf["raw_answer"] = answersdf["answer"]
    answersdf["answer"] = answersdf["raw_answer"].str.split("######", n=0, expand=True)[
        0
    ]
    return answersdf


def qestions_answers_df(
    QA_dict,
    answers_df,
    chat,
    QA_question_col="Q",
    ans_question_col="question",
    QA_base_Q="baseQ",
    QA_answer_col="A",
):
    df_quality_raw = pd.merge(
        pd.DataFrame(QA_dict),
        answers_df,
        left_on=QA_question_col,
        right_on=ans_question_col,
    )
    df_quality_raw = pd.merge(
        df_quality_raw,
        chat.embRetriever.FAQS.loc[:, ["id", "title", "content", "all"]],
        left_on=QA_base_Q,
        right_on="id",
        how="left",
    )
    df_quality_raw.loc[
        (df_quality_raw[QA_answer_col].isna()) | (df_quality_raw[QA_answer_col] == ""),
        QA_answer_col,
    ] = df_quality_raw["content"]
    df_quality_raw.loc[
        df_quality_raw["id"].isna(), QA_answer_col
    ] = "Lamento, mas não consigo responder. Tente colocar a sua questão por outras palavras, por favor."
    return df_quality_raw


def map_assessment_data(
    df_quality_raw,
    question="Q",
    correct_answer="A",
    wrong_answer="WA",
    chat_answer="answer",
):
    assessment_data = []
    for index, item in df_quality_raw.iterrows():
        assessment_data.append(
            {
                "question": item[question],
                "correct_answer": item[correct_answer],
                "wrong_answer": item[wrong_answer]
                if wrong_answer in df_quality_raw.columns
                else "",
                "chat_answer": item[chat_answer],
            }
        )
    return assessment_data
