import re
import json
from core.llm_client import get_llm_client, MODEL_HAIKU

_CHAPTER_RE = re.compile(r"(?:\n\s*)(?:Chapter|CHAPTER)\s+\w+")
_ACT_RE = re.compile(r"(?:\n\s*)ACT [IVX]+")

_DETECT_PROMPT = """\
You are a document structure analyzer. Below is the opening portion of a {domain_label} document.

Your task: identify strings that appear VERBATIM in this text and serve as natural section \
boundaries (e.g. "DIRECT EXAMINATION", "Agenda Item 2:", "CHAPTER THREE").

Rules:
- Only return strings that appear exactly in the text below, character-for-character
- Return between 2 and 10 delimiter strings
- If the text has no consistent section boundaries, return an empty delimiters list
- Respond with valid JSON only — no markdown, no explanation

Format:
{{"doc_type": "<brief type>", "segment_label": "<what each segment represents>", "delimiters": ["<string1>", "<string2>"]}}

TEXT:
"""

_DOMAIN_LABELS: dict[str, str] = {
    "book": "literary",
    "court": "legal / court transcript",
    "meeting": "meeting transcript",
}


def detect_segment_strategy(sample: str, domain: str = "book") -> dict:
    domain_label = _DOMAIN_LABELS.get(domain, domain)
    prompt = _DETECT_PROMPT.format(domain_label=domain_label) + f"\n<document>\n{sample}\n</document>"
    try:
        client = get_llm_client()
        response = client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=200,
            system="",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception:
        return {"doc_type": "unknown", "segment_label": "segment", "delimiters": []}


def split_into_segments(text: str, domain: str = "book") -> list[str]:
    # Fast pre-check: well-formed novel or play — skip LLM entirely.
    # Scan a wider portion than just the first 3000 chars because front
    # matter (title page, blurbs, table of contents) may precede chapter markers.
    head_size = min(3000, len(text))
    sample_head = text[:head_size]
    if (len(_CHAPTER_RE.findall(sample_head)) >= 3
            or len(_ACT_RE.findall(sample_head)) >= 3):
        from core.utils import split_into_chapters
        return split_into_chapters(text)

    # If pre-check failed on head, try the full text — some books have
    # extensive front matter before chapter markers appear.
    if (len(_CHAPTER_RE.findall(text)) >= 2
            or len(_ACT_RE.findall(text)) >= 2):
        from core.utils import split_into_chapters
        return split_into_chapters(text)

    # LLM-detected delimiters
    sample = "\n".join(text.splitlines()[:150])[:4000]
    strategy = detect_segment_strategy(sample, domain=domain)

    delimiters = strategy.get("delimiters") or []
    if delimiters:
        pattern = "|".join(re.escape(d) for d in delimiters)
        segments = re.split(pattern, text)
        segments = [s.strip() for s in segments if s.strip()]
        if len(segments) >= 2:
            return segments

    # Paragraph fallback: segment at paragraph boundaries, merging small
    # paragraphs to avoid producing thousands of tiny segments.
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

    return segments if segments else [text.strip()]
