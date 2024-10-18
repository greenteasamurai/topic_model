# narrative_structure.py

import numpy as np
from collections import Counter

def identify_key_points(chapters):
    polarities = [chapter.mood.vader_sentiment['compound'] for chapter in chapters]
    subjectivities = [chapter.mood.textblob_sentiment['subjectivity'] for chapter in chapters]
    entity_changes = [len(chapter.entities) for chapter in chapters]
    
    key_points = []
    
    # Identify significant polarity shifts
    polarity_diff = np.diff(polarities)
    significant_shifts = np.where(np.abs(polarity_diff) > np.std(polarity_diff))[0]
    for shift in significant_shifts:
        key_points.append((shift + 1, f"Significant mood shift between chapters {shift+1} and {shift+2}"))
    
    # Identify chapters with unusual subjectivity
    subjectivity_mean = np.mean(subjectivities)
    subjectivity_std = np.std(subjectivities)
    for i, subjectivity in enumerate(subjectivities):
        if abs(subjectivity - subjectivity_mean) > 1.5 * subjectivity_std:
            if subjectivity > subjectivity_mean:
                key_points.append((i, f"Chapter {i+1} is unusually subjective"))
            else:
                key_points.append((i, f"Chapter {i+1} is unusually objective"))
    
    # Identify chapters with unusual number of entities
    for i, entities in enumerate(entity_changes):
        if entities > np.mean(entity_changes) + np.std(entity_changes):
            key_points.append((i, f"Chapter {i+1} introduces many new entities"))
    
    # Identify dominant emotions for each chapter
    for i, chapter in enumerate(chapters):
        dominant_emotion = max(chapter.mood.emotions, key=chapter.mood.emotions.get)
        key_points.append((i, f"Chapter {i+1} is dominated by {dominant_emotion} emotion"))
    
    # Identify recurring entities
    all_entities = [entity.name for chapter in chapters for entity in chapter.entities]
    entity_counter = Counter(all_entities)
    recurring_entities = [entity for entity, count in entity_counter.items() if count > len(chapters) / 2]
    if recurring_entities:
        key_points.append((0, f"Recurring entities throughout the book: {', '.join(recurring_entities)}"))
    
    return sorted(key_points)