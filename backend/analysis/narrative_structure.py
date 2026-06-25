import numpy as np
from collections import Counter
from core.data_models import Chapter


def identify_key_points(chapters: list[Chapter]) -> list[tuple[int, str]]:
    polarities = [ch.mood.vader_sentiment["compound"] for ch in chapters]
    subjectivities = [ch.mood.textblob_sentiment["subjectivity"] for ch in chapters]

    key_points: list[tuple[int, str]] = []

    polarity_diff = np.diff(polarities)
    for shift in np.where(np.abs(polarity_diff) > np.std(polarity_diff))[0]:
        key_points.append((int(shift) + 1, f"Significant mood shift between chapters {shift+1} and {shift+2}"))

    subj_mean = float(np.mean(subjectivities))
    subj_std = float(np.std(subjectivities))
    for i, subj in enumerate(subjectivities):
        if abs(subj - subj_mean) > 1.5 * subj_std:
            label = "unusually subjective" if subj > subj_mean else "unusually objective"
            key_points.append((i, f"Chapter {i+1} is {label}"))

    entity_lengths = [len(ch.entities) for ch in chapters]
    mean_e = float(np.mean(entity_lengths))
    std_e = float(np.std(entity_lengths))
    for i, count in enumerate(entity_lengths):
        if count > mean_e + std_e:
            key_points.append((i, f"Chapter {i+1} introduces many new entities"))

    for i, chapter in enumerate(chapters):
        emotions = chapter.mood.emotions
        dominant = max(emotions, key=emotions.get)
        dominant_val = emotions[dominant]
        sorted_vals = sorted(emotions.values(), reverse=True)
        is_strongly_dominant = len(sorted_vals) >= 2 and dominant_val > sorted_vals[1] * 1.5
        if is_strongly_dominant:
            key_points.append((i, f"Chapter {i+1} is dominated by {dominant} emotion"))

    all_names = [e.name for ch in chapters for e in ch.entities]
    recurring = [name for name, cnt in Counter(all_names).items() if cnt > len(chapters) / 2]
    if recurring:
        key_points.append((0, f"Recurring entities throughout the book: {', '.join(recurring)}"))

    return sorted(key_points)
