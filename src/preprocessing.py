import re
import wordninja

def clean_text(text):

    # any lowercase letters immediately followed by
    # capital letters have spaces inserted between them
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)

    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    text = text.replace("\n", " ")

    text = re.sub(r'-\s+', '', text)

    text = " ".join(text.split())

    # saves everything mentioned before references section of the pdf
    text = re.split(r'references|bibliography', text, flags=re.IGNORECASE)[0]

    text = text.lower()

    # fix fully merged lowercase words using wordninja
    words = text.split()
    words = [" ".join(wordninja.split(w)) if len(w) > 20 else w for w in words]
    text = " ".join(words)

    text = re.sub(r'[^\w\s\.-]', '', text)

    return text

def detect_sections(text):
    # regex pattern to detect common sections in research papers, with optional numbering
    pattern = r'((?:\d+[\.\d]*\s+)?(?:abstract|introduction|related work|methodology|methods|results|discussion|conclusion))'

    parts = re.split(pattern, text, flags=re.IGNORECASE)

    sections = {}
    current_section = "preamble"

    for part in parts:
        if re.match(pattern, part, flags=re.IGNORECASE):
            current_section = part.strip().lower()
        else:
            if part.strip():
                sections[current_section] = part.strip()
    
    if list(sections.keys()) == ["preamble"]:
        print("Warning: no sections detected, processing as single document.")

    print(sections)
    return sections