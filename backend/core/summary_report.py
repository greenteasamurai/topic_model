from pathlib import Path
from collections import Counter
from core.data_models import Book

_OUT = Path(__file__).parent.parent.parent / "outputs"
_OUT.mkdir(exist_ok=True)


def generate_summary_report(book: Book, output_dir: Path | None = None) -> None:
    out = output_dir if output_dir is not None else _OUT
    with open(out / "summary_report.md", "w", encoding="utf-8") as f:
        f.write("# Book Analysis Summary Report\n\n")

        avg_sentiment = (sum(ch.mood.vader_sentiment["compound"] for ch in book.chapters) / len(book.chapters)) if book.chapters else 0.0
        f.write("## Overall Statistics\n")
        f.write(f"- Total chapters: {len(book.chapters)}\n")
        f.write(f"- Average sentiment (VADER compound): {avg_sentiment:.2f}\n")
        f.write(f"- Most common entities: {', '.join(e.name for e in book.important_entities[:5])}\n\n")

        overall_emotions: Counter[str] = Counter()
        for ch in book.chapters:
            overall_emotions.update(ch.mood.emotions)
        dominant = max(overall_emotions, key=overall_emotions.get)
        f.write("## Emotional Landscape\n")
        f.write(f"- Dominant emotion: {dominant}\n")
        f.write("- Emotion distribution:\n")
        for emotion, score in sorted(overall_emotions.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {emotion}: {score:.2f}\n")
        f.write("\n")

        f.write("## Theme Development\n### Major Themes\n")
        for theme in book.themes:
            f.write(f"- Topic {theme.id}: {', '.join(theme.keywords)}\n")

        f.write("\n## Key Points in the Narrative\n")
        for _, description in book.key_points:
            f.write(f"- {description}\n")
        f.write("\n")

        if book.character_impacts:
            f.write("## Character Mood Impact\n")
            f.write("Characters ranked by correlation with negative/crisis scenes:\n\n")
            f.write(f"{'Rank':<5} {'Name':<20} {'d':>8} {'Lagged':>8} {'Scenes':>7} {'Crisis':>7}\n")
            f.write("-" * 60 + "\n")
            for rank, imp in enumerate(book.character_impacts, 1):
                f.write(
                    f"{rank:<5} {imp.name:<20} {imp.cohens_d:>+8.3f} "
                    f"{imp.lagged_delta:>+8.3f} {imp.scene_count:>7} {imp.crisis_scene_count:>7}\n"
                )
            f.write("\n")

        f.write("## Chapter-by-Chapter Breakdown\n")
        for ch in book.chapters:
            dominant_emotion = max(ch.mood.emotions, key=ch.mood.emotions.get)
            f.write(f"### Chapter {ch.number}\n")
            f.write(f"- Overall mood: {ch.mood.overall_mood}\n")
            f.write(f"- Sentiment (VADER compound): {ch.mood.vader_sentiment['compound']:.2f}\n")
            f.write(f"- Dominant emotion: {dominant_emotion}\n")
            f.write(f"- Top entities: {', '.join(e.name for e in ch.entities[:3])}\n\n")

    print(f"Summary report generated: {out / 'summary_report.md'}")
