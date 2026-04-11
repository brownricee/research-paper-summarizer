import re
import wordninja

def is_mostly_garbled(text, threshold=0.5):
    words = text.split()
    if not words:
        return False
    garbled = sum(1 for w in words if len(w) > 3 and 
                  not any(c in 'aeiou' for c in w.lower()))
    return garbled / len(words) > threshold

def clean_text(text):

    # any lowercase letters immediately followed by
    # capital letters have spaces inserted between them
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    text = re.sub(r'-\n\s*', '', text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    # fix fully merged lowercase words using wordninja
    words = text.split()
    words = [" ".join(wordninja.split(w)) if len(w) > 15 else w for w in words]
    text = " ".join(words)

    # filter out garbled sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    text = '. '.join(s for s in sentences if not is_mostly_garbled(s))

    text = re.sub(r'[^\w\s\.-]', '', text)

    # filter reversed words from figure visualizations
    words = text.split()
    filtered = []
    for w in words:
        if len(w) > 3:
            rev = w[::-1]
            # if the reversed version splits into fewer pieces, it's a backwards word
            if len(wordninja.split(rev)) < len(wordninja.split(w)):
                continue
        filtered.append(w)
    text = " ".join(filtered)

    text = re.sub(r'\b[a-z]\b', '', text)
    text = " ".join(text.split())

    return text

def detect_sections(text):
    # regex pattern to detect common sections in research papers, with optional numbering
    pattern = r'((?:\d+[\.\d]*\s+)?(?:abstract|introduction|related work|methodology|methods|results|discussion|conclusion|references|bibliography))'

    parts = re.split(pattern, text, flags=re.IGNORECASE)

    sections = {}
    current_section = "preamble"

    for part in parts:
        if re.match(pattern, part, flags=re.IGNORECASE):
            part = re.sub(r'^\d+[\.\d]*\s+', '', part)
            current_section = part.strip().lower()
        else:
            if part.strip():
                if current_section in sections:
                    sections[current_section] += " " + part.strip()
                else:
                    sections[current_section] = part.strip()
    
    if list(sections.keys()) == ["preamble"]:
        print("Warning: no sections detected, processing as single document.")

    return sections