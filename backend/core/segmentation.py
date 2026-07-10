import re
import json
from typing import NamedTuple
from core.llm_client import get_llm_client, MODEL_HAIKU
from core.preclean import preclean_text


class SegmentationResult(NamedTuple):
    segments: list[str]
    characters: list[str]

_CHAPTER_RE = re.compile(r"(?:\n\s*)(?:Chapter|CHAPTER)\s+\w+")
_ACT_RE = re.compile(r"(?:\n\s*)ACT [IVX]+")

_STRUCTURE_PROMPT = """\
You are a literary structure expert. Given a book title and a text sample, \
return the Python regex pattern that matches chapter or section boundaries \
in this specific book, plus a list of the major characters. Use your \
knowledge of the book's known structure and characters. \
If you do not recognize the book, use the text sample for evidence.

The regex will be passed directly to Python's re.split(pattern, text, flags=re.IGNORECASE). \
It should split on the section marker itself (the delimiter is included in the split). \
For example, for a book with chapters labelled "CHAPTER I", "CHAPTER II" etc., \
return the regex: CHAPTER\\s+[IVXLCDM]+

The characters list should include the main named characters, using their \
canonical full names as they appear in the text. Do not include generic titles \
(like "the Friar", "the Nurse") unless that is the only name used. \
Include 5-20 characters.

Rules:
- Return exactly ONE regex pattern that will split the book into its natural sections
- If the book has a hierarchical structure, split at the most granular useful level \
  (chapters or scenes, not sub-headings)
- If you do not recognize the book and the text sample shows no clear structure, \
  return an empty regex string
- Respond with valid JSON only — no markdown, no explanation

Format:
{"regex": "<python regex pattern with proper escaping>", "min_segment_chars": <int or null>, "characters": ["<name1>", "<name2>", ...]}

min_segment_chars: if set, discard any segments shorter than this after splitting \
(set to null if not applicable). Use this when you know the book has short chapter \
title pages (e.g. "THE PRIEST'S TALE:") that should be merged into the next segment."""


def query_book_structure(title: str, sample: str) -> dict:
    prompt = (
        _STRUCTURE_PROMPT
        + f"\n\nBook title: {title}\n\nText sample (opening lines):\n<sample>\n{sample}\n</sample>"
    )
    try:
        client = get_llm_client()
        response = client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=500,
            system="",
            messages=[{"role": "user", "content": prompt}],
            timeout=15,
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception:
        return {"regex": "", "min_segment_chars": None, "characters": []}


def _apply_llm_regex(text: str, pattern: str, min_chars: int | None) -> list[str] | None:
    try:
        segments = re.split(pattern, text, flags=re.IGNORECASE)
    except re.error:
        return None
    segments = [s.strip() for s in segments if s.strip()]
    if len(segments) < 2:
        return None
    if min_chars is not None:
        segments = [s for s in segments if len(s) >= min_chars]
        if len(segments) < 2:
            return None
    small_fraction = sum(1 for s in segments if len(s) < 200) / len(segments)
    if small_fraction > 0.5:
        return None
    return segments


def split_into_segments(text: str, domain: str = "book", title: str = "Unknown") -> SegmentationResult:
    text = preclean_text(text)

    head_size = min(3000, len(text))
    sample_head = text[:head_size]
    if (len(_CHAPTER_RE.findall(sample_head)) >= 3
            or len(_ACT_RE.findall(sample_head)) >= 3):
        from core.utils import split_into_chapters
        return SegmentationResult(split_into_chapters(text), [])

    if (len(_CHAPTER_RE.findall(text)) >= 2
            or len(_ACT_RE.findall(text)) >= 2):
        from core.utils import split_into_chapters
        return SegmentationResult(split_into_chapters(text), [])

    sample = "\n".join(text.splitlines()[:150])[:4000]
    structure = query_book_structure(title, sample)

    regex = structure.get("regex") or ""
    min_chars = structure.get("min_segment_chars")
    characters = structure.get("characters") or []
    if regex:
        llm_segments = _apply_llm_regex(text, regex, min_chars)
        if llm_segments:
            return SegmentationResult(llm_segments, characters)

    paras = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]
    segments: list[str] = []
    buffer = ""
    for para in paras:
        if buffer and len(buffer) + len(para) > 5000:
            segments.append(buffer.strip())
            buffer = para
        else:
            buffer += "\n\n" + para if buffer else para
    if buffer.strip():
        segments.append(buffer.strip())

    if len(segments) < 2:
        segments = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]

    result = segments if segments else [text.strip()]
    return SegmentationResult(result, characters)
