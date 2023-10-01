from mongoengine import connect
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter
from .models import ChatEntry, ChatLike, ChatStats, HourlyMetrics, DailyMetrics


def calculate_metrics(start_time, end_time):
    # Fetch chat entries within the time range and convert them into pandas DataFrame
    chat_entries = list(
        ChatEntry.objects(datetime__gte=start_time, datetime__lt=end_time)
    )
    chat_entries_df = pd.DataFrame.from_records(
        [entry.to_mongo() for entry in chat_entries]
    )

    # Get the list of chat entries' ids
    chat_entry_ids = [str(entry.id) for entry in chat_entries]

    # Fetch all necessary documents based on chat entries' ids and convert them into pandas DataFrames
    chat_likes_df = pd.DataFrame.from_records(
        [like.to_mongo() for like in ChatLike.objects(iteration_id__in=chat_entry_ids)]
    )
    chat_stats_df = pd.DataFrame.from_records(
        [stat.to_mongo() for stat in ChatStats.objects(iteration_id__in=chat_entry_ids)]
    )
    all_cols = [
        "_id",
        "datetime",
        "conversation_id",
        "history",
        "question",
        "summarized_question",
        "base_documents",
        "response",
        "interaction_type",
        "is_avatar",
        "is_avatar_sound",
        "layout_width",
        "layout",
        "time_table",
    ]
    chat_entries_df = chat_entries_df.reindex(columns=all_cols, fill_value=None)
    chat_entries_df["is_avatar"] = chat_entries_df["is_avatar"].fillna(False)
    # Initialize a dictionary to store the counts of each metric
    metrics = Counter()

    # Compute metrics using pandas operations
    metrics["number_of_conversations"] = chat_entries_df["conversation_id"].nunique()
    metrics["number_of_messages"] = chat_entries_df.shape[0]
    metrics["avg_number_of_messages_per_conversation"] = (
        chat_entries_df.groupby("conversation_id").size().mean()
    )

    metrics["voice_input_count"] = (
        chat_entries_df["interaction_type"] == "avatar_mobile_audio_input"
    ).sum()
    metrics["text_input_count"] = (
        chat_entries_df["interaction_type"]
        .isin(["chat_example", "chat_followup", "chat_regular", "chat_retry"])
        .sum()
    )

    metrics["avatar_active_messages"] = chat_entries_df["is_avatar"].sum()
    metrics["avatar_inactive_messages"] = (~chat_entries_df["is_avatar"]).sum()

    metrics["standard_response_count"] = (
        chat_entries_df["response"]
        == "lamento não consigo responder, coloque por outras palavras, por favor"
    ).sum()
    if not chat_likes_df.empty:
        metrics["positive_rated_messages"] = chat_likes_df["like"].sum()
        metrics["negative_rated_messages"] = (~chat_likes_df["like"]).sum()
    if not chat_stats_df.empty:

        metrics["average_response_time"] = chat_stats_df["avatar_tts"].mean()
    # Calculate number_of_clicked_followup_suggested_questions
    metrics["number_of_clicked_followup_suggested_questions"] = (
        chat_entries_df["interaction_type"] == "chat_followup"
    ).sum()

    return metrics


def calculate_hourly_metrics(start_datetime=None):
    if start_datetime is None:
        start_datetime = datetime.now()
    start_time = datetime(
        start_datetime.year,
        start_datetime.month,
        start_datetime.day,
        start_datetime.hour,
    )
    end_time = start_time + timedelta(hours=1)
    metrics = calculate_metrics(start_time, end_time)
    HourlyMetrics.objects(hour_start=start_time).update_one(upsert=True, **metrics)


def calculate_daily_metrics(start_datetime=None):
    if start_datetime is None:
        start_datetime = datetime.now()
    start_time = datetime(start_datetime.year, start_datetime.month, start_datetime.day)
    end_time = start_time + timedelta(days=1)
    metrics = calculate_metrics(start_time, end_time)
    DailyMetrics(day_start=start_time, **metrics).save()

