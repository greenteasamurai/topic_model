import re
import statistics
from core.data_models import Book, CharacterRole
from core.llm_client import get_llm_client, MODEL_HAIKU
from core.utils import turning_point_index

_ROLE_TAXONOMY = {
    "Catalyst": "Triggers a major narrative shift and then exits or fades — high concentration in few scenes near turning points",
    "Gatekeeper": "Controls passage between story phases — appears at arc boundaries, connects otherwise disconnected character clusters",
    "Herald": "Precedes and signals upcoming events — appears shortly before turning points with low personal mood impact",
    "Anchor": "Consistent presence throughout — high scene count, low variance in appearance density, stabilises mood",
    "Mirror": "Reflects or amplifies protagonist state — high co-occurrence with protagonist, parallel mood trajectory",
    "Wildcard": "Sporadic, high-variance presence — unpredictable appearance pattern, outsized impact per scene",
    "Background": "Low impact, consistent presence — filler scenes, little narrative consequence",
}

_ROLE_RULES = [
    {
        "role": "Catalyst",
        "conditions": {
            "_is_articulation_point": ("eq", True),
            "_dwell": ("le", 0.25),
            "_impact_efficiency": ("gt", 0.2),
        },
    },
    {
        "role": "Gatekeeper",
        "conditions": {
            "_is_articulation_point": ("eq", True),
            "_arc_boundary_proximity": ("le", 1),
            "_dwell": ("le", 0.35),
        },
    },
    {
        "role": "Herald",
        "conditions": {
            "_appears_before_turning_point": ("eq", True),
            "_appears_after_turning_point": ("eq", False),
            "_dwell": ("le", 0.2),
            "_scene_count": ("le", 4),
        },
    },
    {
        "role": "Anchor",
        "conditions": {
            "_dwell": ("gt", 0.5),
            "_scene_count": ("gt", 4),
            "_presence_gap_cv": ("le", 1.0),
        },
    },
    {
        "role": "Mirror",
        "conditions": {
            "_dwell": ("gt", 0.3),
            "_max_cooc_dwell": ("gt", 0.5),
            "_mood_correlation": ("gt", 0.25),
        },
    },
    {
        "role": "Wildcard",
        "conditions": {
            "_dwell": ("le", 0.3),
            "_presence_gap_cv": ("gt", 1.5),
            "_impact_efficiency": ("gt", 0.15),
        },
    },
]


def _extract_features(book: Book, entity_name: str) -> dict:
    chapters = book.chapters

    scene_indices: list[int] = []
    for i, ch in enumerate(chapters):
        if any(
            re.search(r"\b" + re.escape(entity_name.lower()) + r"\b", ch.content.lower())
            for e in ch.entities
        ):
            scene_indices.append(i)

    first_scene = scene_indices[0] if scene_indices else 0
    last_scene = scene_indices[-1] if scene_indices else 0
    total_scenes = len(chapters)
    scene_count = len(scene_indices)
    dwell = scene_count / total_scenes if total_scenes else 0.0

    gaps: list[int] = []
    for i in range(1, len(scene_indices)):
        gaps.append(scene_indices[i] - scene_indices[i - 1])
    gap_cv = (statistics.stdev(gaps) / statistics.mean(gaps)) if len(gaps) >= 2 and statistics.mean(gaps) > 0 else 0.0

    scores = [ch.mood.vader_sentiment["compound"] for ch in chapters]
    tp_idx = turning_point_index(scores)

    appears_before_tp = tp_idx is not None and first_scene < tp_idx
    appears_after_tp = tp_idx is not None and last_scene > tp_idx

    impact: float = 0.0
    e_count: float = 0.0
    for imp in book.character_impacts:
        if imp.name.lower() == entity_name.lower():
            impact = abs(imp.lagged_delta)
            e_count = imp.scene_count
            break
    impact_efficiency = impact / e_count if e_count > 0 else 0.0

    is_articulation = False
    if book.articulation:
        is_articulation = any(
            ap.name.lower() == entity_name.lower()
            for ap in book.articulation.articulation_points
        )

    arc_labels = {ch.number: ch.arc_label for ch in chapters}
    min_boundary_dist = total_scenes
    for idx in scene_indices:
        if idx > 0 and arc_labels.get(idx - 1, -1) != arc_labels.get(idx, -1):
            min_boundary_dist = 0
            break
        if idx + 1 in arc_labels and arc_labels.get(idx, -1) != arc_labels.get(idx + 1, -1):
            min_boundary_dist = 0
            break

    scene_scores_present = [scores[i] for i in scene_indices]
    mood_correlation = statistics.stdev(scene_scores_present) / (abs(statistics.mean(scene_scores_present)) + 1) if len(scene_scores_present) >= 2 else 0.0

    return {
        "scene_count": scene_count,
        "dwell": dwell,
        "presence_gap_cv": gap_cv,
        "appears_before_turning_point": appears_before_tp,
        "appears_after_turning_point": appears_after_tp,
        "impact_efficiency": impact_efficiency,
        "is_articulation_point": is_articulation,
        "arc_boundary_proximity": min_boundary_dist,
        "mood_correlation": mood_correlation,
    }


def _check_rules(features: dict) -> tuple[str | None, float, list[str]]:
    for rule in _ROLE_RULES:
        reasons: list[str] = []
        met = 0
        total = len(rule["conditions"])
        for key, (op, val) in rule["conditions"].items():
            fv = features.get(key)
            if fv is None:
                continue
            if op == "gt" and fv > val:
                met += 1
                reasons.append(f"{key}={fv:.3f} > {val}")
            elif op == "le" and fv <= val:
                met += 1
                reasons.append(f"{key}={fv:.3f} <= {val}")
            elif op == "eq" and fv == val:
                met += 1
                reasons.append(f"{key}={fv} == {val}")
        if met == total:
            return rule["role"], 1.0, reasons
    return None, 0.0, []


def _llm_classify(book: Book, entity_name: str) -> tuple[str, float, list[str]]:
    try:
        chapters = book.chapters
        scene_nums = []
        for i, ch in enumerate(chapters):
            if any(e.name.lower() == entity_name.lower() for e in ch.entities):
                scene_nums.append(str(i + 1))

        impact_line = ""
        for imp in book.character_impacts:
            if imp.name.lower() == entity_name.lower():
                impact_line = f"Cohen's d={imp.cohens_d:.3f}, lagged_delta={imp.lagged_delta:.3f}, scenes={imp.scene_count}"
                break

        arc_map: dict[int, list[int]] = {}
        for ch in chapters:
            if ch.arc_label >= 0:
                arc_map.setdefault(ch.arc_label, []).append(ch.number)
        arc_str = "; ".join(
            f"arc {k}: scenes {min(v)}-{max(v)}"
            for k, v in sorted(arc_map.items())
        ) if arc_map else "none"

        roles_list = "\n".join(
            f"- {key}: {desc}"
            for key, desc in _ROLE_TAXONOMY.items()
        )

        prompt = f"""You are a narrative role classifier. Classify the character "{entity_name}" in "{book.title}" into exactly one of the following roles:

{roles_list}

Character stats:
  Appears in scenes: {', '.join(scene_nums) if scene_nums else 'unknown'}
  {impact_line}
  Arc structure: {arc_str}

Respond with valid JSON only. Format:
{{"role": "<role_name>", "reasons": ["<reason1>", "<reason2>"]}}"""

        client = get_llm_client()
        response = client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=300,
            system="",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        import json
        parsed = json.loads(raw)
        role = parsed.get("role", "Background")
        reasons = parsed.get("reasons", ["LLM classification"])
        return role, 0.7, reasons
    except Exception:
        return "Background", 0.3, ["LLM fallback failed"]


def compute_character_roles(book: Book, use_llm_fallback: bool = True) -> list[CharacterRole]:
    seen = set()
    results: list[CharacterRole] = []
    for entity in book.important_entities:
        if entity.name.lower() in seen:
            continue
        seen.add(entity.name.lower())

        features = _extract_features(book, entity.name)
        role, confidence, reasons = _check_rules(features)

        if role is None and use_llm_fallback:
            role, confidence, reasons = _llm_classify(book, entity.name)

        if role is None:
            role = "Background"
            confidence = 0.2
            reasons = ["No rules matched; no LLM fallback"]

        results.append(CharacterRole(
            name=entity.name,
            role=role,
            confidence=confidence,
            reasons=reasons,
        ))

    return results
