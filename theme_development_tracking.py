# theme_development_tracking.py

from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from data_models import Topic

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens if token not in STOPWORDS and token.isalpha()]

def track_theme_development(chapters, num_topics=5):
    preprocessed_chapters = [preprocess_text(chapter) for chapter in chapters]
    
    dictionary = corpora.Dictionary(preprocessed_chapters)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_chapters]
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    
    theme_development = []
    for i, chapter_bow in enumerate(corpus):
        chapter_topics = lda_model.get_document_topics(chapter_bow)
        theme_development.append([
            Topic(id=topic_id, keywords=get_topic_keywords(lda_model, topic_id), weight=weight)
            for topic_id, weight in sorted(chapter_topics, key=lambda x: x[1], reverse=True)
        ])
    
    return lda_model, theme_development

def get_topic_keywords(lda_model, topic_id, num_words=5):
    return [word for word, _ in lda_model.show_topic(topic_id, num_words)]

def print_topics(lda_model, num_words=5):
    return [(idx, get_topic_keywords(lda_model, idx, num_words)) for idx in range(lda_model.num_topics)]

def analyze_theme_shifts(theme_development):
    theme_shifts = []
    prev_dominant_theme = max(theme_development[0], key=lambda x: x.weight)
    
    for i in range(1, len(theme_development)):
        current_dominant_theme = max(theme_development[i], key=lambda x: x.weight)
        if current_dominant_theme.id != prev_dominant_theme.id:
            theme_shifts.append({
                'chapter': i + 1,
                'from': prev_dominant_theme.id,
                'to': current_dominant_theme.id
            })
        prev_dominant_theme = current_dominant_theme
    
    return theme_shifts