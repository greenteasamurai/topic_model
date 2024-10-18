import re
import spacy

nlp = spacy.load("en_core_web_sm")

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_chapters(text):
    # First, try to split by "Chapter" or "CHAPTER"
    chapters = re.split(r'(?:\n\s*)(?:Chapter|CHAPTER)\s+\w+', text)
    
    # If that results in only one or two segments, try splitting by Act and Scene
    if len(chapters) <= 2:
        acts = re.split(r'(?:\n\s*)ACT [IVX]+', text)
        chapters = []
        for act in acts[1:]:  # Skip the first split as it's before ACT I
            scenes = re.split(r'(?:\n\s*)SCENE [IVX]+', act)
            chapters.extend(scenes)
    
    # Remove empty chapters and strip whitespace
    chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
    
    return chapters

def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]

def get_noun_chunks(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]