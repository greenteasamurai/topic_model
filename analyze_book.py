from utils import read_file, split_into_chapters
from entity_extraction import extract_important_entities
from entity_network import process_chapter_entities
from mood_analysis import get_chapter_mood
from visualization import visualize_mood_flow, visualize_entity_flow
from narrative_structure import identify_key_points
from summary_report import generate_summary_report
from topic_modeling import extract_topics, get_chapter_topics

def analyze_book(file_path):
    full_text = read_file(file_path)
    chapters = split_into_chapters(full_text)
    
    print(f"Number of chapters/scenes detected: {len(chapters)}")
    
    important_entities = extract_important_entities(chapters)
    
    results = []
    all_chapter_moods = []
    all_chapter_entities = []
    
    # Extract topics for the entire book
    lda_model, dictionary, topic_names = extract_topics(chapters)
    
    for i, chapter in enumerate(chapters):
        print(f"Processing chapter/scene {i+1}...")
        
        chapter_entities = extract_important_entities([chapter])
        all_chapter_entities.append(chapter_entities)
        
        chapter_mood = get_chapter_mood(chapter)
        all_chapter_moods.append(chapter_mood)
        
        entity_graph = process_chapter_entities(i+1, [entity for entity, _ in chapter_entities])
        
        chapter_topics = get_chapter_topics(lda_model, dictionary, chapter, topic_names)
        
        results.append({
            'chapter': i+1,
            'entities': chapter_entities,
            'mood': chapter_mood,
            'entity_graph': entity_graph,
            'topics': chapter_topics
        })
    
    if len(chapters) > 1:
        visualize_mood_flow(all_chapter_moods)
        visualize_entity_flow(all_chapter_entities, [e for e, _ in important_entities])
        key_points = identify_key_points(results)
        generate_summary_report(results, key_points)
    else:
        print("Not enough chapters/scenes to perform comprehensive analysis.")
    
    return {
        'chapters': chapters,
        'important_entities': important_entities,
        'results': results,
        'all_chapter_moods': all_chapter_moods,
        'all_chapter_entities': all_chapter_entities,
        'key_points': key_points if len(chapters) > 1 else []
    }
