import re

__all__ = ["preclean_text"]

_URL_PATTERN = re.compile(r"https?://\S+")

_TRANSCRIBER_NOTES = re.compile(
    r"(?i)^\s*(?:Transcriber|\u2019s?\s+Note[s]?|Typographical\s+Errors)\s*:?\s*$",
    re.MULTILINE,
)

_SIGNATURE_BLOCKS = re.compile(
    r"(?i)^\s*(?:Transcriber|Produced\s+by|Proofread\s+by|Digitized\s+by|Scanned\s+by"
    r"|HTML\s+version|PG\s+Editions|Prepared\s+by)[\w\s,.'\u2019\-]*$",
    re.MULTILINE,
)

_OCR_FRAGMENTS = re.compile(r"^\s*[\u2018\u2019\u201c\u201d\u2026\u00b6\u2020\u2021\u00a7]\s*$", re.MULTILINE)

_REPEATED_JUNK = re.compile(r"^[^\w\s]{20,}\s*$", re.MULTILINE)

_LONG_REPEATING = re.compile(r"(.)\1{30,}")

_GUTENBERG_START = re.compile(
    r"\*\*\*\s*START\s*OF\s*(?:THE|THIS)\s*PROJECT\s*GUTENBERG\s*(?:EBOOK|ETEXT).*?\*\*\*",
    re.IGNORECASE,
)
_GUTENBERG_END = re.compile(
    r"\*\*\*\s*END\s*OF\s*(?:THE|THIS)\s*PROJECT\s*GUTENBERG\s*(?:EBOOK|ETEXT).*",
    re.IGNORECASE,
)


def _strip_gutenberg(text: str) -> str:
    m_start = _GUTENBERG_START.search(text)
    if m_start:
        text = text[m_start.end():]
    m_end = _GUTENBERG_END.search(text)
    if m_end:
        text = text[:m_end.start()]
    return text


def _strip_frontmatter(text: str) -> str:
    m = _GUTENBERG_START.search(text[:20000])
    if m:
        return text[m.end():]
    return text


def _strip_backmatter(text: str) -> str:
    m = _GUTENBERG_END.search(text[-50000:])
    if m:
        return text[:len(text) - 50000 + m.start()]
    return text


def preclean_text(text: str) -> str:
    t = _strip_gutenberg(text[:])

    t = _TRANSCRIBER_NOTES.sub("", t)
    t = _URL_PATTERN.sub("", t)
    t = _SIGNATURE_BLOCKS.sub("", t)
    t = _OCR_FRAGMENTS.sub("", t)
    t = _REPEATED_JUNK.sub("", t)
    t = _LONG_REPEATING.sub("", t)

    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"^\n+", "", t)
    t = re.sub(r"\n+$", "", t)

    return t
