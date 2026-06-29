from collections import Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from core.utils import extract_entities as _extract_entities
from core.data_models import Entity

_LABEL_TO_TYPE: dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Place",
    "WORK_OF_ART": "Work of Art",
}

_FALSE_POSITIVES_BY_DOMAIN: dict[str, set[str]] = {
    "book": {
        "tis", "twas", "thou", "thee", "thy", "thine", "thyself", "ye",
        "hath", "doth", "dost", "hast", "wilt", "shalt",
        "farewell", "nay", "marry", "faith", "madam", "ho", "shall",
        "sweet", "speak", "enter", "exit", "exeunt",
        "lady", "friar", "sir", "lord",
    },
    "court": {
        "honor", "counsel", "plaintiff", "defendant", "witness",
        "testimony", "objection", "sustained", "overruled",
    },
    "meeting": {
        "agenda", "attendees", "action", "item", "minutes", "follow-up",
    },
}

_CENTRALITY_ALPHA: float = 0.45


def extract_important_entities(chapters: list[str], top_n: int = 10, domain: str = "book") -> list[Entity]:
    false_positives = _FALSE_POSITIVES_BY_DOMAIN.get(domain, set())
    all_entity_names: list[str] = []
    entity_labels: dict[str, list[str]] = {}
    chapter_entity_texts: list[str] = []
    cooccurrence: nx.Graph = nx.Graph()

    for chapter in chapters:
        entities = [
            (name, label)
            for name, label in _extract_entities(chapter)
            if name.lower() not in false_positives
        ]

        seen_lower = list(dict.fromkeys(name.lower() for name, _ in entities))
        for i, n1 in enumerate(seen_lower):
            for n2 in seen_lower[i + 1:]:
                if cooccurrence.has_edge(n1, n2):
                    cooccurrence[n1][n2]["weight"] += 1
                else:
                    cooccurrence.add_edge(n1, n2, weight=1)

        weighted_names: list[str] = []
        for j, (name, label) in enumerate(entities):
            boost = 1 + (0.1 if j < max(len(entities) // 10, 1) else 0)
            weighted_names.extend([name] * int(boost * 10))
            entity_labels.setdefault(name.lower(), []).append(label)

        all_entity_names.extend(weighted_names)
        chapter_entity_texts.append(" ".join(name for name, _ in entities))

    if not all_entity_names:
        return []

    entity_counts: Counter[str] = Counter(all_entity_names)

    # Only run TF-IDF if at least one chapter had entities
    has_any_entity_text = any(t.strip() for t in chapter_entity_texts)
    if has_any_entity_text:
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(chapter_entity_texts)
            tfidf_scores: dict[str, float] = {
                feature: float(tfidf_matrix[:, i].sum())
                for i, feature in enumerate(vectorizer.get_feature_names_out())
            }
        except ValueError:
            tfidf_scores = {}
    else:
        tfidf_scores = {}

    combined_scores: dict[str, float] = {
        name: count * (1 + tfidf_scores.get(name.lower(), 0.0))
        for name, count in entity_counts.items()
    }

    merged: dict[str, dict] = {}
    for name, score in combined_scores.items():
        key = name.lower()
        if key in merged:
            merged[key]["score"] += score
            merged[key]["variants"].append(name)
        else:
            merged[key] = {"score": score, "variants": [name]}

    if cooccurrence.number_of_nodes() > 1:
        centrality = nx.betweenness_centrality(cooccurrence, weight="weight", normalized=True)
        max_freq = max(data["score"] for data in merged.values()) or 1.0
        max_c = max(centrality.values()) or 1.0
        for key in merged:
            freq_norm = merged[key]["score"] / max_freq
            c_norm = centrality.get(key, 0.0) / max_c
            merged[key]["score"] = (1 - _CENTRALITY_ALPHA) * freq_norm + _CENTRALITY_ALPHA * c_norm

    top_entries = sorted(merged.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]

    result: list[Entity] = []
    for key, data in top_entries:
        canonical = Counter(data["variants"]).most_common(1)[0][0]
        label_counts = Counter(entity_labels.get(key, []))
        label = label_counts.most_common(1)[0][0] if label_counts else "Unknown"
        result.append(Entity(
            name=canonical,
            count=int(data["score"] * 10000),
            entity_type=_LABEL_TO_TYPE.get(label, "Unknown"),
        ))

    return result
