import uuid
from mongoengine import (
    Document,
    StringField,
    DateTimeField,
    ListField,
    BooleanField,
    ReferenceField,
    DictField,
    EmbeddedDocument,
    EmbeddedDocumentField,
    FloatField,
    IntField,
)
from datetime import datetime


class BaseDocument(Document):
    id = StringField(primary_key=True)
    title = StringField(required=True)
    content = StringField()
    type = StringField()
    all = StringField()
    source = StringField()
    topic = StringField()
    title_embs = StringField()
    embeddings = ListField(FloatField())
    all_embs = StringField()
    embeddings_qa = ListField(FloatField())
    topic_number = IntField()
    category_id = IntField()
    category_name = StringField()
    sub_category_id = IntField()
    sub_category_name = StringField()
    meta = {"allow_inheritance": True}


class DocumentWithSimilarity(EmbeddedDocument):
    document = ReferenceField(BaseDocument)
    similarity = BooleanField(default=False)


class ChatEntry(Document):
    datetime = DateTimeField(default=datetime.utcnow)
    conversation_id = StringField(default=str(uuid.uuid4()))
    history = ListField(DictField())
    question = StringField(required=True)
    summarized_question = StringField(required=True)
    base_documents = ListField(EmbeddedDocumentField(DocumentWithSimilarity))
    response = StringField(required=True)
    interaction_type = StringField()
    is_avatar = BooleanField()
    is_avatar_sound = BooleanField()
    layout_width = StringField()
    layout = StringField()
    time_table = DictField()


class ChatLike(Document):
    datetime = DateTimeField(default=datetime.utcnow)
    conversation_id = StringField(default=str(uuid.uuid4()))
    iteration_id = StringField(required=True)
    like = BooleanField(required=True)
    reason = StringField(required=False)


class ChatStats(Document):
    datetime = DateTimeField(default=datetime.utcnow)
    conversation_id = StringField()
    iteration_id = StringField()
    backend = FloatField()
    avatar_tts = FloatField()
    avatar = FloatField()


class MetricsBase(Document):
    meta = {'allow_inheritance': True}

    number_of_conversations = IntField(default=0)
    number_of_users = IntField(default=0)
    number_of_messages = IntField(default=0)
    average_response_time = FloatField(default=0.0)
    number_of_clicked_initial_suggested_questions = IntField(default=0)
    number_of_clicked_followup_suggested_questions = IntField(default=0)
    positive_rated_messages = IntField(default=0)
    negative_rated_messages = IntField(default=0)
    # messages with "lamento não consigo responder,
    # coloque por outras palavras, por favor"
    standard_response_count = IntField(default=0)
    voice_input_count = IntField(default=0)
    text_input_count = IntField(default=0)
    avatar_active_messages = IntField(default=0)
    avatar_inactive_messages = IntField(default=0)
    avg_number_of_messages_per_conversation = FloatField(default=0.0)


class HourlyMetrics(MetricsBase):
    hour_start = DateTimeField(required=True)
    meta = {
        'indexes': [
            'hour_start',
        ],
    }


class DailyMetrics(MetricsBase):
    day_start = DateTimeField(required=True)


class QnA(Document):
    """Question-Answer pairs"""
    id = StringField(primary_key=True)
    title = StringField(required=True)
    id_forum = IntField()  # ID of the Fórum topic
    content = StringField()
    update_ = IntField()  # `update` is a reserved keyword, hence the `_`
    url = StringField()  # URL of the NOS Fórum page
    category_id = IntField()
    category_name = StringField()
    sub_category_id = IntField()
    sub_category_name = StringField()  

class SubCategory(EmbeddedDocument):
    """Sub-category within a category"""
    sub_category_id = IntField(required=True)
    sub_category_name = StringField(required=True)

class Category(Document):
    """Category"""
    category_id = IntField(primary_key=True)
    category_name = StringField(required=True)
    sub_categories = ListField(EmbeddedDocumentField(SubCategory))

class BestAnswer(Document):
    """BestAnswer"""
    id = StringField(primary_key=True)
    title = StringField(required=True)
    id_forum = IntField()  # ID of the Fórum best answer
    url = StringField()# URL of the NOS Fórum page
    question = StringField()
    answer = StringField()
    category_id = IntField()
    category_name = StringField()
    sub_category_id = IntField()
    sub_category_name = StringField() 


class LogInfo(EmbeddedDocument):
    logger_name = StringField()
    level = StringField()
    module = StringField()
    line = IntField()
    timestamp = DateTimeField()
    message = StringField()


class ConversationLogs(Document):
    _id = StringField(primary_key=True)
    conversation_id = StringField()
    logs = ListField(EmbeddedDocumentField(LogInfo))
