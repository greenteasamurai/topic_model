import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils import read_file, split_into_chapters
from preprocess import preprocess_chapter
from topic_modeling import extract_topics
from visualization import visualize_topics_by_chapter
from entity_network import process_chapter_entities
from entity_extraction import extract_important_entities

def main():
    file_path = "books/Dracula.txt"
    full_text = read_file(file_path)
    chapters = split_into_chapters(full_text)
    
    lda_model, dictionary, topic_names = extract_topics(chapters)

    visualize_topics_by_chapter(lda_model, dictionary, chapters, topic_names)
    
    # Extract important entities for the entire book
    important_entities = extract_important_entities(chapters)
    print("Top entities for the entire book:", important_entities)
    
    results = []
    for i, chapter in enumerate(chapters):
        print(f"Processing chapter {i+1}...")
        
        # Extract important entities for this chapter
        chapter_entities = extract_important_entities([chapter])
        
        bow = dictionary.doc2bow(preprocess_chapter(chapter))
        topics = lda_model.get_document_topics(bow)
        
        # Convert topic IDs to names
        named_topics = [(topic_names[topic_id], weight) for topic_id, weight in topics]
        
        # Process entity network for the chapter
        entity_graph = process_chapter_entities(i+1, [entity for entity, _ in chapter_entities])
        
        results.append({
            'chapter': i+1,
            'entities': chapter_entities,
            'topics': named_topics,
            'entity_graph': entity_graph
        })
    
    # Print or further process results as needed
    for result in results:
        print(f"\nChapter {result['chapter']}:")
        print(f"  Entities: {result['entities']}")
        print(f"  Topics: {result['topics']}")
        print(f"  Entity Network: {result['entity_graph'].number_of_edges()} connections")

if __name__ == "__main__":
    main()