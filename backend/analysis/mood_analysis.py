import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from analysis.custom_emotion import custom_get_emotion
from core.data_models import Mood, VaderSentiment, TextBlobSentiment

nltk.download("vader_lexicon", quiet=True)


def get_chapter_mood(chapter: str) -> Mood:
    sia = SentimentIntensityAnalyzer()
    raw = sia.polarity_scores(chapter)
    vader_scores: VaderSentiment = {
        "neg": raw["neg"],
        "neu": raw["neu"],
        "pos": raw["pos"],
        "compound": raw["compound"],
    }

    blob = TextBlob(chapter)
    textblob_sentiment: TextBlobSentiment = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }

    return Mood(
        overall_mood=_determine_overall_mood(vader_scores["compound"]),
        vader_sentiment=vader_scores,
        textblob_sentiment=textblob_sentiment,
        emotions=dict(custom_get_emotion(chapter)),
    )


def _determine_overall_mood(compound_score: float) -> str:
    if compound_score >= 0.05:
        return "positive"
    if compound_score <= -0.05:
        return "negative"
    return "neutral"
