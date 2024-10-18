from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import extract_entities as extract_entities_base, nlp
from data_models import Entity


def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]
    
    # Filter out common false positives
    filtered_entities = [entity for entity in entities if entity[0].lower() not in ['tis', 'twas']]

    return filtered_entities

def extract_important_entities(chapters, top_n=10):
    all_entities = []
    chapter_entities = []

    for i, chapter in enumerate(chapters):
        entities = extract_entities_base(chapter)
        
        # Apply position-based weighting
        weighted_entities = []
        for j, entity in enumerate(entities):
            weight = 1 + (0.1 if j < len(entities) // 10 else 0)  # 10% boost for entities in the first 10% of the chapter
            weighted_entities.extend([entity] * int(weight * 10))  # Multiply by 10 to keep it as integer
        
        all_entities.extend(weighted_entities)
        chapter_entities.append(" ".join(set(entities)))  # For TF-IDF

    # Frequency counting
    entity_counts = Counter(all_entities)

    # TF-IDF calculation
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chapter_entities)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_scores = {}
    for i, feature in enumerate(feature_names):
        tfidf_scores[feature] = sum(tfidf_matrix[:, i].toarray()[0])

    # Combine frequency and TF-IDF scores
    combined_scores = {}
    for entity, count in entity_counts.items():
        tfidf_score = tfidf_scores.get(entity, 0)
        combined_scores[entity] = count * (1 + tfidf_score)  # Multiply frequency by (1 + TF-IDF) to boost important entities

    # Combine entities based on case-insensitivity
    case_insensitive_scores = {}
    for entity, score in combined_scores.items():
        key = entity.lower()
        if key in case_insensitive_scores:
            case_insensitive_scores[key]['score'] += score
            case_insensitive_scores[key]['variants'].append(entity)
        else:
            case_insensitive_scores[key] = {'score': score, 'variants': [entity]}

    # Choose the most common variant for each entity
    final_entities = []
    for key, data in case_insensitive_scores.items():
        most_common_variant = Counter(data['variants']).most_common(1)[0][0]
        final_entities.append((most_common_variant, data['score']))

    # Get top N entities
    top_entities = sorted(final_entities, key=lambda x: x[1], reverse=True)[:top_n]

    return [
        Entity(name=entity, count=count, entity_type=determine_entity_type(entity))
        for entity, count in top_entities[:top_n]
    ]

def determine_entity_type(entity):
    # Implement logic to determine if the entity is a person, place, or organization
    # This could involve using NER from spaCy or a custom method
    if entity[1] == "PERSON":
        return "Person"
    elif entity[1] == "ORG":
        return "Organization"
    elif entity[1] == "GPE":
        return "Place"
    elif entity[1] == "WORK_OF_ART":
        return "Work of Art"
    else:
        return "Unknown"
    pass