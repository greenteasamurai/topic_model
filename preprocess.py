from utils import preprocess_text, extract_entities


def preprocess_chapter(chapter):
    return preprocess_text(chapter)

# This function is now redundant, but kept for backwards compatibility
def extract_entities(chapter):
    return extract_entities(chapter)