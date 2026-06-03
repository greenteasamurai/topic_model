import re
import spacy

nlp = spacy.load("en_core_web_sm")


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_into_chapters(text: str) -> list[str]:
    chapters = re.split(r"(?:\n\s*)(?:Chapter|CHAPTER)\s+\w+", text)
    if len(chapters) <= 2:
        acts = re.split(r"(?:\n\s*)ACT [IVX]+", text)
        chapters = []
        for act in acts[1:]:
            scenes = re.split(r"(?:\n\s*)SCENE [IVX]+", act)
            chapters.extend(scenes)
    return [chapter.strip() for chapter in chapters if chapter.strip()]


def preprocess_text(text: str) -> list[str]:
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]


def extract_entities(text: str) -> list[tuple[str, str]]:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]


def get_noun_chunks(text: str) -> list[str]:
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]
