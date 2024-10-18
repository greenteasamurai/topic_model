import spacy
import networkx as nx
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

def analyze_character_relationships(chapters):
    character_graph = nx.Graph()
    character_mentions = defaultdict(int)
    
    for chapter in chapters:
        doc = nlp(chapter)
        
        # Extract named entities that are likely characters (PERSON)
        characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        # Count character mentions
        for character in characters:
            character_mentions[character] += 1
        
        # Create edges between characters mentioned in the same chapter
        for i in range(len(characters)):
            for j in range(i+1, len(characters)):
                if character_graph.has_edge(characters[i], characters[j]):
                    character_graph[characters[i]][characters[j]]['weight'] += 1
                else:
                    character_graph.add_edge(characters[i], characters[j], weight=1)
    
    # Remove characters with few mentions (likely not main characters)
    min_mentions = sum(character_mentions.values()) / len(character_mentions) / 2
    for character, mentions in list(character_mentions.items()):
        if mentions < min_mentions:
            if character in character_graph:
                character_graph.remove_node(character)
            del character_mentions[character]
    
    return character_graph, character_mentions

def get_main_characters(character_mentions, top_n=5):
    return sorted(character_mentions.items(), key=lambda x: x[1], reverse=True)[:top_n]

def get_character_relationships(character_graph, character):
    if character not in character_graph:
        return []
    
    relationships = []
    for neighbor in character_graph.neighbors(character):
        weight = character_graph[character][neighbor]['weight']
        relationships.append((neighbor, weight))
    
    return sorted(relationships, key=lambda x: x[1], reverse=True)