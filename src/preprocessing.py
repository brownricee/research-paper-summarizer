import pdfplumber
import spacy
import re

def clean_text(text):
    text = text.replace("\n", " ")

    text = re.sub(r'-\s+', '', text)

    text = " ".join(text.split())

    # saves everything mentioned before references section of the pdf
    text = re.split(r'references|bibliography', text, flags=re.IGNORECASE)[0]

    text = text.lower()
    text = re.sub(r'[^\w\s\.-]', '', text)

    return text

def detect_sections(text):
    pattern = r'((?:\d+[\.\d]*\s+)?(?:abstract|introduction|related work|methodology|methods|results|discussion|conclusion))'

    parts = re.split(pattern, text, flags=re.IGNORECASE)

    sections = {}
    current_section = "preamble"

    for part in parts:
        if re.match(pattern, part, flags=re.IGNORECASE):
            current_section = part.strip().lower()
        else:
            sections[current_section] = part.strip()
    
    return sections