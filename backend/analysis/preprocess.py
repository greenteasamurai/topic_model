from core.utils import preprocess_text


def preprocess_chapter(chapter: str) -> list[str]:
    return preprocess_text(chapter)
