from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from analysis.preprocess import preprocess_chapter
from core.utils import preprocess_text
from core.data_models import Topic

_ARCHAIC = {
    "thou", "thee", "thy", "thine", "thyself", "tis", "twas",
    "hath", "doth", "dost", "hast", "wilt", "shalt", "art", "ye",
}
_STOPWORDS = list(ENGLISH_STOP_WORDS | _ARCHAIC)


def extract_topics(
    chapters: list[str],
    num_topics: int | None = None,
) -> tuple[BERTopic, list[Topic]]:
    if len(chapters) < 3:
        return _dummy_topics()
    if num_topics is None:
        num_topics = min(20, max(3, len(chapters) // 5))
    vectorizer = CountVectorizer(stop_words=_STOPWORDS)
    model = BERTopic(
        nr_topics=num_topics,
        language="english",
        min_topic_size=2,
        vectorizer_model=vectorizer,
    )
    model.fit_transform(chapters)

    topics = [
        Topic(
            id=topic_id,
            keywords=[word for word, _ in words[:3]],
            weight=0.0,
        )
        for topic_id, words in model.get_topics().items()
        if topic_id != -1
    ]

    return model, topics


def _dummy_topics() -> tuple[BERTopic, list[Topic]]:
    model = BERTopic(nr_topics=1)
    return model, []


def get_chapter_topics(
    model: BERTopic,
    chapter: str,
    topics: list[Topic],
) -> list[Topic]:
    if not topics:
        return []
    topics_by_id = {t.id: t for t in topics}
    topic_distr, _ = model.approximate_distribution([chapter], window=4, stride=1)

    return [
        Topic(id=tid, keywords=topics_by_id[tid].keywords, weight=float(weight))
        for tid, weight in enumerate(topic_distr[0])
        if tid in topics_by_id and float(weight) > 0.01
    ]
