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