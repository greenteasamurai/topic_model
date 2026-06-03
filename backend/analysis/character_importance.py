import re
import math
import statistics
from core.data_models import Book, CharacterImpact

_CRISIS_THRESHOLD = -0.3


def _present_in(name: str, content: str) -> bool:
    return bool(re.search(r"\b" + re.escape(name.lower()) + r"\b", content.lower()))


def _cohens_d(present: list[float], absent: list[float]) -> float:
    if len(present) < 2 or len(absent) < 2:
        return 0.0
    n1, n2 = len(present), len(absent)
    s1, s2 = statistics.stdev(present), statistics.stdev(absent)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0.0:
        return 0.0
    return (statistics.mean(absent) - statistics.mean(present)) / pooled


def compute_character_mood_impact(book: Book) -> list[CharacterImpact]:
    impacts: list[CharacterImpact] = []

    for entity in book.important_entities:
        if entity.entity_type == "Place":
            continue

        present_scores: list[float] = []
        absent_scores: list[float] = []
        lagged_deltas: list[float] = []
        crisis_count = 0

        for i, chapter in enumerate(book.chapters):
            score = chapter.mood.vader_sentiment["compound"]
            if _present_in(entity.name, chapter.content):
                present_scores.append(score)
                if score < _CRISIS_THRESHOLD:
                    crisis_count += 1
                if i < len(book.chapters) - 1:
                    next_score = book.chapters[i + 1].mood.vader_sentiment["compound"]
                    lagged_deltas.append(next_score - score)
            else:
                absent_scores.append(score)

        if not present_scores:
            continue

        impacts.append(CharacterImpact(
            name=entity.name,
            presence_avg=statistics.mean(present_scores),
            absence_avg=statistics.mean(absent_scores) if absent_scores else 0.0,
            mood_delta=(statistics.mean(absent_scores) if absent_scores else 0.0)
                       - statistics.mean(present_scores),
            cohens_d=_cohens_d(present_scores, absent_scores),
            lagged_delta=statistics.mean(lagged_deltas) if lagged_deltas else 0.0,
            scene_count=len(present_scores),
            crisis_scene_count=crisis_count,
        ))

    return sorted(impacts, key=lambda x: x.cohens_d, reverse=True)
