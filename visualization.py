import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_chapter
from topic_modeling import extract_topics

def visualize_topics_by_chapter(lda_model, dictionary, chapters, topic_names):
    num_topics = lda_model.num_topics
    topic_distribution = np.zeros((len(chapters), num_topics))
    
    for i, chapter in enumerate(chapters):
        bow = dictionary.doc2bow(preprocess_chapter(chapter))
        chapter_topics = lda_model.get_document_topics(bow)
        for topic_id, weight in chapter_topics:
            topic_distribution[i, topic_id] = weight
    
    plt.figure(figsize=(15, 10))
    im = plt.imshow(topic_distribution, aspect='auto', cmap='YlOrRd')
    plt.imshow(topic_distribution, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Topic Weight')
    plt.xlabel('Topics')
    plt.ylabel('Chapters')
    plt.title('Topic Distribution Across Chapters')
    
    plt.xticks(range(num_topics), topic_names, rotation=45, ha='right') 
    plt.tight_layout()
    plt.savefig('topic_distribution.png')
    plt.close()