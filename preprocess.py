import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_chapter(chapter):
    doc = nlp(chapter)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

def extract_entities(chapter):
    doc = nlp(chapter)
    return [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]