from collections import defaultdict
import anthropic
from core.data_models import Book, CharacterImpact

_CLIENT = anthropic.Anthropic()
_MODEL = "claude-sonnet-4-6"


def _quadrant(imp: CharacterImpact) -> str:
    if imp.cohens_d > 0 and imp.lagged_delta < 0:
        return "active catalyst"
    if imp.cohens_d > 0 and imp.lagged_delta >= 0:
        return "tragic presence"
    if imp.cohens_d <= 0 and imp.lagged_delta < 0:
        return "hidden driver"
    return "stabiliser"


def _build_prompt(book: Book, domain: str) -> tuple[str, str]:
    char_rows = "\n".join(
        f"  {imp.name:<18} d={imp.cohens_d:>+6.3f}  lag={imp.lagged_delta:>+6.3f}"
        f"  n={imp.scene_count}  crisis={imp.crisis_scene_count}  [{_quadrant(imp)}]"
        for imp in book.character_impacts
    )

    topic_rows = "\n".join(
        f"  Topic {t.id}: {', '.join(t.keywords)}"
        for t in book.themes
    )

    arc_scenes: dict[int, list[int]] = defaultdict(list)
    for ch in book.chapters:
        if ch.arc_label >= 0:
            arc_scenes[ch.arc_label].append(ch.number)
    arc_rows = "\n".join(
        f"  Arc {k}: scenes {min(v)}–{max(v)}"
        for k, v in sorted(arc_scenes.items())
    ) or "  No distinct arcs detected"

    scores = [ch.mood.vader_sentiment["compound"] for ch in book.chapters]
    if len(scores) > 1:
        diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        tp = diffs.index(min(diffs))
        tp_names = ", ".join(e.name for e in book.chapters[tp].entities[:3])
        turning_point = f"Between scenes {tp + 1} and {tp + 2} (featuring {tp_names})"
    else:
        turning_point = "Insufficient data"

    avg_score = sum(scores) / len(scores) if scores else 0.0

    domain_frame = {
        "book": "literary work",
        "court": "oral argument transcript",
        "meeting": "meeting transcript",
    }.get(domain, "text")

    static = f"""You are a rigorous narrative analyst. The following quantitative analysis was run on the {domain_frame} "{book.title}".

METRIC DEFINITIONS
  Cohen's d (d): how strongly a character's presence correlates with darker scenes.
    Positive = character appears during the work's darker passages.
  Lagged delta (lag): average sentiment change in the scene AFTER the character appears.
    Negative = their appearance tends to precede a sentiment drop (causal signal).
  Quadrant classification:
    active catalyst  — in dark scenes AND causes subsequent drops
    tragic presence  — in dark scenes, mood stabilises afterward
    hidden driver    — not in dark scenes but consistently precedes drops
    stabiliser       — lighter scenes, mood holds or improves after appearances

CHARACTER MOOD IMPACT
{char_rows}

TOPICS DISCOVERED BY BERTOPIC
{topic_rows}

NARRATIVE ARCS (HDBSCAN scene clustering)
{arc_rows}

SENTIMENT TURNING POINT
  {turning_point}

OVERALL SENTIMENT
  Mean: {avg_score:+.3f}  |  Range: {min(scores):+.3f} to {max(scores):+.3f}"""

    task = f"""Write 3–5 paragraphs of subtextual analysis of "{book.title}". Do not merely restate the numbers — interpret them. Address:

1. What the character quadrant distribution reveals about the work's dramatic structure.
2. Which characters the data marks as most consequential, and whether this confirms or complicates standard critical readings.
3. What the topic clusters and arc structure say about thematic development.
4. Any surprising or counter-intuitive findings in the data.

Cite specific metric values where they illuminate your argument. Be direct and precise."""

    return static, task


def analyze_with_llm(book: Book, domain: str = "book") -> str:
    static_context, task = _build_prompt(book, domain)

    response = _CLIENT.messages.create(
        model=_MODEL,
        max_tokens=1500,
        system=[
            {
                "type": "text",
                "text": "You are a rigorous narrative analyst who integrates quantitative evidence with close reading. You write in clear, direct prose — no bullet points, no hedging.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": static_context,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": task,
                    },
                ],
            }
        ],
    )

    return response.content[0].text
