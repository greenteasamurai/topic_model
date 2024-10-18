import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from custom_emotion import custom_get_emotion
from data_models import Mood

nltk.download('vader_lexicon')

def analyze_mood(text):
    # VADER sentiment analysis
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)
    
    # TextBlob sentiment analysis
    blob = TextBlob(text)
    textblob_sentiment = blob.sentiment
    
    # Emotion recognition using custom function
    emotions = custom_get_emotion(text)
    
    return {
        'vader_sentiment': vader_scores,
        'textblob_sentiment': {
            'polarity': textblob_sentiment.polarity,
            'subjectivity': textblob_sentiment.subjectivity
        },
        'emotions': emotions
    }

def get_chapter_mood(chapter):
    mood_analysis = analyze_mood(chapter)
    
    return Mood(
        overall_mood=determine_overall_mood(mood_analysis['vader_sentiment']['compound']),
        vader_sentiment=mood_analysis['vader_sentiment'],
        textblob_sentiment=mood_analysis['textblob_sentiment'],
        emotions=dict(mood_analysis['emotions'])
    )

def determine_overall_mood(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'