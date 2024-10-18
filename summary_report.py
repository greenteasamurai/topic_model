# summary_report.py

from collections import Counter

def generate_summary_report(book):
    with open('summary_report.md', 'w') as f:
        f.write("# Book Analysis Summary Report\n\n")
        
        f.write("## Overall Statistics\n")
        f.write(f"- Total chapters: {len(book.chapters)}\n")
        f.write(f"- Average sentiment (VADER compound): {sum(chapter.mood.vader_sentiment['compound'] for chapter in book.chapters) / len(book.chapters):.2f}\n")
        f.write(f"- Most common entities: {', '.join(entity.name for entity in book.important_entities[:5])}\n\n")
        
        f.write("## Emotional Landscape\n")
        overall_emotions = Counter()
        for chapter in book.chapters:
            overall_emotions.update(chapter.mood.emotions)
        dominant_emotion = max(overall_emotions, key=overall_emotions.get)
        f.write(f"- Dominant emotion: {dominant_emotion}\n")
        f.write("- Emotion distribution:\n")
        for emotion, score in sorted(overall_emotions.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {emotion}: {score:.2f}\n")
        f.write("\n")
        
        f.write("## Theme Development\n")
        f.write("### Major Themes\n")
        for theme in book.themes:
            f.write(f"- Topic {theme.id}: {', '.join(theme.keywords)}\n")
        
        f.write("\n## Key Points in the Narrative\n")
        for chapter, description in book.key_points:
            f.write(f"- {description}\n")
        f.write("\n")
        
        f.write("## Chapter-by-Chapter Breakdown\n")
        for chapter in book.chapters:
            f.write(f"### Chapter {chapter.number}\n")
            f.write(f"- Overall mood: {chapter.mood.overall_mood}\n")
            f.write(f"- Sentiment (VADER compound): {chapter.mood.vader_sentiment['compound']:.2f}\n")
            f.write(f"- Dominant emotion: {max(chapter.mood.emotions, key=chapter.mood.emotions.get)}\n")
            f.write(f"- Top entities: {', '.join(entity.name for entity in chapter.entities[:3])}\n")
            f.write("\n")

    print("Summary report generated: summary_report.md")