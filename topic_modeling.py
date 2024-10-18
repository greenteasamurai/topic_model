# topic_modeling.py

from gensim import corpora
from gensim.models import LdaModel, HdpModel
from preprocess import preprocess_chapter
from utils import preprocess_text
from data_models import Topic

def extract_topics(chapters, num_topics=5, hierarchical=False):
    preprocessed_chapters = [preprocess_chapter(chapter) for chapter in chapters]
    dictionary = corpora.Dictionary(preprocessed_chapters)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_chapters]
    
    if hierarchical:
        model = HdpModel(corpus=corpus, id2word=dictionary)
        topics = []
        for topic_id in range(model.get_topics().shape[0]):
            top_words = model.show_topic(topic_id, topn=3)
            topic_name = ', '.join([word for word, _ in top_words])
            topics.append(Topic(id=topic_id, keywords=[word for word, _ in top_words], weight=0))  # Weight is not applicable for HDP
    else:
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15)
        topics = []
        for topic_id in range(num_topics):
            top_words = model.show_topic(topic_id, topn=3)
            topic_name = ', '.join([word for word, _ in top_words])
            topics.append(Topic(id=topic_id, keywords=[word for word, _ in top_words], weight=0))  # Weight will be set when applied to specific text
    
    return model, dictionary, topics

def get_chapter_topics(model, dictionary, chapter, topics):
    bow = dictionary.doc2bow(preprocess_text(chapter))
    chapter_topics = model.get_document_topics(bow)
    return [Topic(id=topic_id, keywords=topics[topic_id].keywords, weight=weight) for topic_id, weight in chapter_topics]

def get_hierarchical_topics(model, dictionary, chapter, topics):
    bow = dictionary.doc2bow(preprocess_text(chapter))
    chapter_topics = model.get_document_topics(bow)
    return [Topic(id=topic_id, keywords=topics[topic_id].keywords, weight=weight) for topic_id, weight in chapter_topics if topic_id < len(topics)]