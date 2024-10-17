import gensim.downloader as api

# Load pre-trained word embeddings
word_vectors = api.load("glove-wiki-gigaword-100")

def find_related_words(word, topn=5):
    try:
        similar_words = word_vectors.most_similar(word, topn=topn)
        return [word for word, _ in similar_words]
    except KeyError:
        return []

def expand_topics_with_related_words(topics, topn=3):
    expanded_topics = []
    for topic in topics:
        expanded_topic = topic.copy()
        for word in topic:
            related_words = find_related_words(word, topn=topn)
            expanded_topic.extend(related_words)
        expanded_topics.append(list(set(expanded_topic)))  # Remove duplicates
    return expanded_topics
