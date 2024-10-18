# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter

def visualize_mood_flow(chapters):
    compound_scores = [chapter.mood.vader_sentiment['compound'] for chapter in chapters]
    
    plt.figure(figsize=(15, 10))
    plt.plot(range(1, len(chapters) + 1), compound_scores, marker='o')
    plt.title('Mood Flow Across Chapters')
    plt.xlabel('Chapter')
    plt.ylabel('VADER Sentiment Compound Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mood_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_emotion_distribution(chapters):
    emotion_counts = Counter()
    for chapter in chapters:
        emotion_counts.update(chapter.mood.emotions)
    
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=emotions, y=counts)
    plt.title('Overall Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Total Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('emotion_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_character_network(important_entities):
    G = nx.Graph()
    for entity in important_entities:
        G.add_node(entity.name, weight=entity.count)
    
    # Add edges (this is a simplification, you may want to adjust this logic)
    for i, entity1 in enumerate(important_entities):
        for entity2 in important_entities[i+1:]:
            G.add_edge(entity1.name, entity2.name)
    
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=[G.nodes[node]['weight'] * 100 for node in G.nodes], 
            font_size=10, font_weight='bold')
    
    plt.title('Character Relationship Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('character_network.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_entity_flow(chapters, important_entities):
    top_entities = [entity.name for entity in important_entities[:10]]  # Top 10 entities
    
    entity_matrix = []
    for chapter in chapters:
        chapter_entities = [entity.name for entity in chapter.entities]
        chapter_vector = [1 if entity in chapter_entities else 0 for entity in top_entities]
        entity_matrix.append(chapter_vector)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(entity_matrix, cmap='YlOrRd', yticklabels=top_entities)
    plt.title('Entity Flow Across Chapters')
    plt.xlabel('Chapter')
    plt.ylabel('Entity')
    plt.tight_layout()
    plt.savefig('entity_flow.png', dpi=300, bbox_inches='tight')
    plt.close()