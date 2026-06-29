import re
from pathlib import Path
import chardet
import pdfplumber
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def read_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(pages)).strip()


def read_file(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        return read_pdf(file_path)
    raw = Path(file_path).read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    return raw.decode(encoding, errors="replace")


def split_into_chapters(text: str) -> list[str]:
    chapters = re.split(r"(?:\n\s*)(?:Chapter|CHAPTER)\s+\w+", text)
    if len(chapters) <= 2:
        acts = re.split(r"(?:\n\s*)ACT [IVX]+", text)
        chapters = []
        for act in acts[1:]:
            scenes = re.split(r"(?:\n\s*)SCENE [IVX]+", act)
            chapters.extend(scenes)

    stripped = [chapter.strip() for chapter in chapters if chapter.strip()]

    # Merge segments that are too small to be real chapters — likely
    # table-of-contents entries or single-line section headers.
    merged: list[str] = []
    for s in stripped:
        txt = s.strip()
        if not txt:
            continue
        if merged and (len(txt.splitlines()) <= 2 and len(txt) < 200):
            merged[-1] += "\n\n" + txt
        else:
            merged.append(txt)
    return merged if merged else stripped


def preprocess_text(text: str) -> list[str]:
    doc = _get_nlp()(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]


def extract_entities(text: str) -> list[tuple[str, str]]:
    doc = _get_nlp()(text)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]


def get_noun_chunks(text: str) -> list[str]:
    doc = _get_nlp()(text)
    return [chunk.text for chunk in doc.noun_chunks]


def turning_point_index(compound_scores: list[float]) -> int | None:
    """Return 0-based index of the scene just before the steepest sentiment drop, or None."""
    if len(compound_scores) < 2:
        return None
    diffs = [compound_scores[i + 1] - compound_scores[i] for i in range(len(compound_scores) - 1)]
    return diffs.index(min(diffs))
