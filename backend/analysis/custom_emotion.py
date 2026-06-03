import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

EMOTION_LEXICON: dict[str, set[str]] = {
    "joy": {"happy", "joyful", "delighted", "cheerful", "excited", "glad", "pleased", "thrilled", "elated", "jubilant"},
    "sadness": {"sad", "unhappy", "depressed", "gloomy", "miserable", "sorrowful", "dejected", "heartbroken", "downcast", "grieving"},
    "anger": {"angry", "furious", "enraged", "irritated", "annoyed", "mad", "frustrated", "irate", "outraged", "resentful"},
    "fear": {"afraid", "scared", "frightened", "terrified", "anxious", "worried", "panicked", "nervous", "startled", "fearful"},
    "surprise": {"surprised", "astonished", "amazed", "shocked", "stunned", "astounded", "bewildered", "dumbfounded", "flabbergasted", "awestruck"},
}

_STOP_WORDS = set(stopwords.words("english"))


def custom_get_emotion(text: str) -> list[tuple[str, float]]:
    tokens = [t for t in word_tokenize(text.lower()) if t not in _STOP_WORDS]

    emotion_counts: Counter[str] = Counter()
    for token in tokens:
        for emotion, words in EMOTION_LEXICON.items():
            if token in words:
                emotion_counts[emotion] += 1

    total = sum(emotion_counts.values())
    if total > 0:
        percentages = {emotion: emotion_counts[emotion] / total for emotion in EMOTION_LEXICON}
    else:
        percentages = {emotion: 0.0 for emotion in EMOTION_LEXICON}

    return sorted(percentages.items(), key=lambda x: x[1], reverse=True)
