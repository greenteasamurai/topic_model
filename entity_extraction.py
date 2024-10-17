import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def extract_important_entities(chapters, top_n=10):
    all_entities = []
    chapter_entities = []

    for i, chapter in enumerate(chapters):
        doc = nlp(chapter)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "LOC", "ORG"]]
        
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

    # Get top N entities
    top_entities = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return top_entities