# main.py

from data_models import Book, Chapter, Mood, Entity, Topic
from mood_analysis import get_chapter_mood
from entity_extraction import extract_important_entities
from topic_modeling import extract_topics, get_chapter_topics
from narrative_structure import identify_key_points
from utils import read_file, split_into_chapters
from visualization import visualize_mood_flow, visualize_emotion_distribution, visualize_character_network, visualize_entity_flow
from summary_report import generate_summary_report

def analyze_book(file_path):
    full_text = read_file(file_path)
    chapters = split_into_chapters(full_text)
    
    important_entities = extract_important_entities(chapters)
    lda_model, dictionary, topics = extract_topics(chapters)
    
    book_chapters = []
    for i, chapter_text in enumerate(chapters):
        mood = get_chapter_mood(chapter_text)
        entities = extract_important_entities([chapter_text])
        chapter_topics = get_chapter_topics(lda_model, dictionary, chapter_text, topics)
        
        book_chapters.append(Chapter(
            number=i+1,
            content=chapter_text,
            mood=mood,
            entities=entities,
            topics=chapter_topics
        ))
    
    key_points = identify_key_points(book_chapters)
    
    book = Book(
        title=extract_title(file_path),
        chapters=book_chapters,
        important_entities=important_entities,
        themes=topics,
        key_points=key_points
    )
    
    # Generate visualizations
    visualize_mood_flow(book.chapters)
    visualize_emotion_distribution(book.chapters)
    visualize_character_network(book.important_entities)
    visualize_entity_flow(book.chapters, book.important_entities)
    
    # Generate summary report
    generate_summary_report(book)
    
    return book

def extract_title(file_path):
    return file_path.split("/")[-1].replace(".txt", "") # Extract title from file path
    pass

def main():
    file_path = "books/RomeoAndJuliet.txt"  # Update this path as needed
    analysis_result = analyze_book(file_path)
    
    # Print some results for verification
    print("\nSample results:")
    for i, chapter in enumerate(analysis_result.chapters[:3]):  # Print first 3 chapters' results
        print(f"\nChapter {chapter.number}:")
        print(f"Overall mood: {chapter.mood.overall_mood}")
        print(f"VADER sentiment (compound): {chapter.mood.vader_sentiment['compound']:.2f}")
        print(f"Top entities: {', '.join(e.name for e in chapter.entities[:3])}")
        print(f"Top topics: {', '.join(f'Topic {t.id}' for t in chapter.topics[:3])}")

if __name__ == "__main__":
    main()