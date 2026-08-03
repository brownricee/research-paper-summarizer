import re
import wordninja
from collections import Counter

CITATION_RE = re.compile(r'\[\s*\d+(?:\s*[,;–—-]\s*\d+)*\s*\]')

def is_mostly_garbled(text, threshold=0.5):
    words = text.split()
    if not words:
        return False
    garbled = sum(1 for w in words if len(w) > 3 and 
                  not any(c in 'aeiou' for c in w.lower()))
    return garbled / len(words) > threshold

def clean_text(text):
    # Cleans URLs
    text = re.sub(r'https?://\S+(?:\s+\S+/\S+)*|www\.\S+(?:\s+\S+/\S+)*', '', text)
    # Strips citation markers
    text = re.sub(CITATION_RE, '', text)
    # Any lowercase letters immediately followed by
    # capital letters have spaces inserted between them
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    text = re.sub(r'(\w+)-\s+(\w+)', _rejoin_hyphen, text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    # Prevents decimal values from being destroyed.
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)

    # Fix fully merged lowercase words using wordninja
    words = text.split()
    words = [" ".join(wordninja.split(w)) if len(w) > 15 else w for w in words]
    text = " ".join(words)

    # Filter out garbled sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    text = " ".join(s for s in sentences if not is_mostly_garbled(s) and not is_table_junk(s) and not has_contact_info(s))

    text = re.sub(r'[^\w\s\.,:%()=+<>-]', '', text)

    # Filter reversed words from figure visualizations
    words = text.split()
    filtered = []
    for w in words:
        if len(w) > 3:
            rev = w[::-1]
            # If the reversed version splits into fewer pieces, it's a backwards word
            if len(wordninja.split(rev)) < len(wordninja.split(w)):
                continue
        filtered.append(w)
    text = " ".join(filtered) 

    return text

def detect_sections(text, words=None):
    if not words:
        return detect_sections_by_regex(text)

    body_size = get_body_font_size(words)

    # Regroup words into separate lines
    lines = []
    current, current_top = [], None
    for w in words:
        top = round(w['top'], 1)
        if current and top != current_top:
            lines.append(current)
            current = []
        current.append(w)
        current_top = top
    if current:
        lines.append(current)
    
    # Walk through all lines and split on headings (when font size is big and words are bolded)
    sections = {}
    current_section = "preamble"
    # Break on these words if detected in a section heading.
    TERMINAL_SECTION_RE = re.compile(
        r'^\s*(?:\d+[\.\d]*\s+)?'
        r'(references|bibliography|acknowledge?ments?|funding'
        r'|author contributions?|conflicts? of interest)\b',
        re.IGNORECASE,
    )
    for line in lines:
        line_text = " ".join(w['text'] for w in line).strip()

        if TERMINAL_SECTION_RE.search(line_text):
            break
        if is_heading(line, body_size):
            name = re.sub(r'^\d+[\.\d]*\s+', '', line_text).lower()
            current_section = name
        elif not is_body_line(line, body_size):
            continue
        else:
            sections[current_section] = (sections.get(current_section, "") + " " + line_text).strip()

    if list(sections.keys()) == ["preamble"]:
        print("Warning: no headings detected, processing as single document")
    
    return sections

def detect_sections_by_regex(text):
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

def get_body_font_size(words):
    sizes = Counter()

    for w in words:
        size = w.get("size")
        if size is None:
            continue
        sizes[round(size, 1)] += len(w["text"])
    if not sizes:
        return None
    return sizes.most_common(1)[0][0]

def is_bold(fontname):
    # Pdfplumber fontnames embed the weight, e.g. "ABCDEF+NimbusRomNo9L-Medi"
    # or "...-Bold". Guard against None in case a word has no fontname.
    return bool(fontname) and "bold" in fontname.lower()

def is_heading(line_words, body_size, max_heading_words=8, size_tolerance=0.5):
    if not line_words:
        return False

    text = " ".join(w["text"] for w in line_words).strip()
    if not text:
        return False

    # Must contain a letter - rejects page numbers, equation/figure
    # labels, and other digit/symbol-only lines that may be bold or large.
    if not any(ch.isalpha() for ch in text):
        return False

    # Headings are short - rejects a bold or large emphasized
    # sentence sitting inside a paragraph.
    if len(text.split()) > max_heading_words:
        return False

    # Headings don't end like a running sentence (a trailing period,
    # comma, etc.). This also filters most figure/table captions.
    if text[-1] in ".!?,;":
        return False

    # Font signal, aggregated across the line: use the largest word size (a
    # leading section number can render slightly smaller than the words) and
    # treat the line as bold if any of its words are bold.
    sizes = [round(w["size"], 1) for w in line_words if w.get("size") is not None]
    line_bold = any(is_bold(w.get("fontname")) for w in line_words)

    if body_size is None:
        # No body baseline to compare against — fall back to weight alone.
        return line_bold

    larger_than_body = bool(sizes) and max(sizes) >= body_size + size_tolerance
    return larger_than_body or line_bold

def is_table_junk(sentence):
    letters = sum(c.isalpha() for c in sentence)
    non_letters = sum(c.isdigit() or (not c.isalnum() and not c.isspace()) for c in sentence)
    ratio = non_letters / max(1, letters + non_letters)

    # Used to drop sentences that mainly contain numbers that could confuse the summarization model
    # later in the pipeline
    return ratio > 0.3

def has_contact_info(sentence):
    # Checks for emails which is an @ sign, a dotted domain, or a space-split domain
    if re.search(r'@|\.(?:com|edu|org|net)\b|\b\w+ (?:com|edu|org|net)\b', sentence, re.IGNORECASE):
        return True

    # Author/affiliation blocks get shredded into many 1-2 char tokens, so this check gets rid of that
    tokens = sentence.split()
    if tokens and sum(1 for t in tokens if len(t) <= 2) / len(tokens) > 0.5:
        return True

    return False

def _rejoin_hyphen(m):
    merged = m.group(1) + m.group(2)

    # if word-ninja sees one word, return merged
    if len(wordninja.split(merged)) == 1:
        return merged
    else:
        return m.group(0)

def is_body_line(line_words, body_size, size_tolerance=0.5):
    if body_size is None:
        return True

    sizes = [round(w["size"], 1) for w in line_words if w.get("size") is not None]
    if not sizes:
        return True

    return max(sizes) >= body_size - size_tolerance