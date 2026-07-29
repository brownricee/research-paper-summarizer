import re
import pdfplumber
from src.preprocessing import clean_text
from src.preprocessing import detect_sections

def getTextFromPDF(pdf_path):
  text = ""

  try:
    with pdfplumber.open(pdf_path) as pdf:
      all_words = []
      for page in pdf.pages:
        words = page.extract_words(x_tolerance=2, keep_blank_chars=True, extra_attrs=["fontname", "size"])
        all_words.extend(words)
        # group words into lines by their vertical position
        lines = {}
        for w in words:
          top = round(w['top'], 1)
          if top not in lines:
            lines[top] = []
          lines[top].append(w['text'])
        page_text = "\n".join(" ".join(lines[t]) for t in sorted(lines))
        text += page_text + "\n"
  except FileNotFoundError:
    print("The file you tried to provide does not exist.")


  # Cut the raw text at the first References/Bibliography header so nothing
  # after it ever enters section detection.
  refs_match = re.search(r'\n\s*(references|bibliography)\s*\n', text, flags=re.IGNORECASE)
  if refs_match:
    text = text[:refs_match.start()]
  
  sections = detect_sections(text, all_words)
  sections.pop("preamble", None)
  sections.pop("references", None)
  sections.pop("bibliography", None)
  for name in list(sections.keys()):
    if re.search(r"acknowledg|appendix|author", name, flags=re.IGNORECASE):
      del sections[name]

  for key, value in sections.items():
    value = clean_text(value)
    sections[key] = value

  return sections