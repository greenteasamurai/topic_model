import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Simple emotion lexicon
EMOTION_LEXICON = {
    'joy': {'happy', 'joyful', 'delighted', 'cheerful', 'excited', 'glad', 'pleased', 'thrilled', 'elated', 'jubilant'},
    'sadness': {'sad', 'unhappy', 'depressed', 'gloomy', 'miserable', 'sorrowful', 'dejected', 'heartbroken', 'downcast', 'grieving'},
    'anger': {'angry', 'furious', 'enraged', 'irritated', 'annoyed', 'mad', 'frustrated', 'irate', 'outraged', 'resentful'},
    'fear': {'afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'panicked', 'nervous', 'startled', 'fearful'},
    'surprise': {'surprised', 'astonished', 'amazed', 'shocked', 'stunned', 'astounded', 'bewildered', 'dumbfounded', 'flabbergasted', 'awestruck'}
}

def custom_get_emotion(text):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Count emotions
    emotion_counts = Counter()
    for token in tokens:
        for emotion, words in EMOTION_LEXICON.items():
            if token in words:
                emotion_counts[emotion] += 1


    # Calculate percentages
    total = sum(emotion_counts.values())
    if total > 0:
        emotion_percentages = {emotion: count / total for emotion, count in emotion_counts.items()}
    else:
        emotion_percentages = {emotion: 0 for emotion in EMOTION_LEXICON.keys()}
    
    # Ensure all emotions are present in the result
    for emotion in EMOTION_LEXICON.keys():
        if emotion not in emotion_percentages:
            emotion_percentages[emotion] = 0

    ranked_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_emotions