from pathlib import Path
from core.data_models import Book, Chapter
from analysis.mood_analysis import get_chapter_mood
from analysis.entity_extraction import extract_important_entities
from analysis.topic_modeling import extract_topics, get_chapter_topics
from analysis.narrative_structure import identify_key_points
from analysis.arc_detection import detect_arcs
from analysis.character_importance import compute_character_mood_impact
from analysis.role_detection import compute_character_roles
from analysis.articulation_analysis import analyze_articulation_points, analyze_temporal_articulation
from core.utils import read_file
from core.segmentation import split_into_segments
from core.visualization import (
    visualize_mood_flow,
    visualize_emotion_distribution,
    visualize_character_network,
    visualize_entity_flow,
    visualize_character_mood_impact,
    visualize_character_impact_scatter,
    visualize_articulation_structure,
)
from core.summary_report import generate_summary_report

_BOOKS_DIR = Path(__file__).parent.parent / "books"


def _subchunk_large_segments(segments: list[str], max_chars: int = 200000) -> list[str]:
    """Subdivide any segment exceeding max_chars at paragraph boundaries."""
    result: list[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            result.append(seg)
        else:
            import re
            paras = [s.strip() for s in re.split(r"\n{2,}", seg) if s.strip()]
            chunk = ""
            for para in paras:
                if chunk and len(chunk) + len(para) > max_chars:
                    result.append(chunk.strip())
                    chunk = para
                else:
                    chunk += "\n\n" + para if chunk else para
            if chunk.strip():
                result.append(chunk.strip())
    return result


def _cap_segments(segments: list[str], max_segments: int = 100) -> list[str]:
    """Cap the number of segments to prevent extreme runtime on long books.
    Merges segments pairwise until under the limit."""
    if len(segments) <= max_segments:
        return segments
    result = list(segments)
    while len(result) > max_segments:
        merged: list[str] = []
        for i in range(0, len(result), 2):
            if i + 1 < len(result):
                merged.append(result[i] + "\n\n" + result[i + 1])
            else:
                merged.append(result[i])
        result = merged
    return result


def analyze_book(text: str, title: str = "Unknown", output_dir: Path | None = None, domain: str = "book") -> Book:
    result = split_into_segments(text, domain, title=title)
    chapters = result.segments
    known_characters = result.characters
    chapters = _subchunk_large_segments(chapters)
    chapters = _cap_segments(chapters)
    important_entities = extract_important_entities(chapters, top_n=15, domain=domain, known_characters=known_characters or None)
    topic_model, themes = extract_topics(chapters)

    book_chapters: list[Chapter] = []
    for i, chapter_text in enumerate(chapters):
        book_chapters.append(Chapter(
            number=i + 1,
            content=chapter_text,
            mood=get_chapter_mood(chapter_text),
            entities=extract_important_entities([chapter_text], domain=domain, known_characters=known_characters or None),
            topics=get_chapter_topics(topic_model, chapter_text, themes),
        ))

    arc_labels = detect_arcs([ch.content for ch in book_chapters])
    for chapter, label in zip(book_chapters, arc_labels):
        chapter.arc_label = label

    key_points = identify_key_points(book_chapters)
    book = Book(
        title=title,
        chapters=book_chapters,
        important_entities=important_entities,
        themes=themes,
        key_points=key_points,
    )

    book.character_impacts = compute_character_mood_impact(book)
    book.character_roles = compute_character_roles(book)
    book.articulation = analyze_articulation_points(book, min_cooccurrence=1)
    book.articulation_weighted = analyze_articulation_points(book, min_cooccurrence=2)
    book.articulation_temporal = analyze_temporal_articulation(book, min_cooccurrence=2)

    visualize_mood_flow(book.chapters, output_dir)
    visualize_emotion_distribution(book.chapters, output_dir)
    visualize_character_network(book.chapters, book.important_entities, output_dir)
    visualize_entity_flow(book.chapters, book.important_entities, output_dir)
    visualize_character_mood_impact(book.character_impacts, output_dir)
    visualize_character_impact_scatter(book.character_impacts, output_dir)
    visualize_articulation_structure(book, book.articulation, output_dir)
    generate_summary_report(book, output_dir)

    return book


def main() -> None:
    file_path = _BOOKS_DIR / "RomeoAndJuliet.txt"
    title = file_path.stem
    book = analyze_book(read_file(str(file_path)), title=title)

    art = book.articulation
    if art:
        print("\nArticulation points (unweighted):")
        if art.articulation_points:
            for ap in art.articulation_points:
                print(f"  {ap.name} — splits into {ap.components_after_removal} groups:")
                for c in ap.clusters:
                    print(f"    [{', '.join(c)}]")
        else:
            print("  None — network is biconnected")

    art_w = book.articulation_weighted
    if art_w:
        print("\nArticulation points (min 2 shared scenes):")
        if art_w.articulation_points:
            for ap in art_w.articulation_points:
                print(f"  {ap.name} — splits into {ap.components_after_removal} groups:")
        else:
            print("  None — network is biconnected")

    if book.articulation_temporal:
        print("\nTemporal articulation (min 2 co-occurrences):")
        for snap in book.articulation_temporal:
            aps = ", ".join(snap.articulation_point_names) or "none"
            print(f"  Through scene {snap.up_to_scene:>3}: [{aps}]")

    print("\nCharacter mood impact (top 10 by Cohen's d):")
    print(f"  {'Name':<18} {'d':>7}  {'lagged':>8}  {'present_avg':>12}  {'n':>4}  {'crisis':>6}")
    for imp in book.character_impacts[:10]:
        print(
            f"  {imp.name:<18} {imp.cohens_d:>+7.3f}  {imp.lagged_delta:>+8.3f}"
            f"  {imp.presence_avg:>+12.3f}  {imp.scene_count:>4}  {imp.crisis_scene_count:>6}"
        )

    print()
    for chapter in book.chapters[:3]:
        top_topics = sorted(chapter.topics, key=lambda t: t.weight, reverse=True)[:3]
        topic_strs = [f"Topic {t.id} [{t.weight:.3f}] ({', '.join(t.keywords)})" for t in top_topics]
        print(f"\nChapter {chapter.number} (arc {chapter.arc_label}):")
        print(f"Overall mood: {chapter.mood.overall_mood}")
        print(f"VADER sentiment (compound): {chapter.mood.vader_sentiment['compound']:.2f}")
        print(f"Top entities: {', '.join(e.name for e in chapter.entities[:3])}")
        print(f"Top topics: {', '.join(topic_strs)}")


if __name__ == "__main__":
    main()
