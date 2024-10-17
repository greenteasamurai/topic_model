from gensim import corpora
from gensim.models import LdaModel
from preprocess import preprocess_chapter

def extract_topics(chapters, num_topics=5):
    preprocessed_chapters = [preprocess_chapter(chapter) for chapter in chapters]
    dictionary = corpora.Dictionary(preprocessed_chapters)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_chapters]
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    
    # Extract topic names
    topic_names = []
    for topic_id in range(num_topics):
        top_words = lda_model.show_topic(topic_id, topn=3)
        topic_name = ', '.join([word for word, _ in top_words])
        topic_names.append(topic_name)
    
    return lda_model, dictionary, topic_names