def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_chapters(text, chapter_delimiter="\n\nCHAPTER"):
    return text.split(chapter_delimiter)