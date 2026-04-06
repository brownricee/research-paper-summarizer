import re
import pdfplumber
from src.preprocessing import clean_text
from src.preprocessing import detect_sections

def getTextFromPDF(pdf_path):
  text = ""
  try:
    with pdfplumber.open(pdf_path) as pdf:
      for page in pdf.pages:
        words = page.extract_words(x_tolerance=2, keep_blank_chars=True)
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

  # strip references/bibliography before section detection
  text = re.split(r'\nreferences\s*\n|\nbibliography\s*\n', text, flags=re.IGNORECASE)[0]

  sections = detect_sections(text)

  for key, value in sections.items():
    value = clean_text(value)
    sections[key] = value

  return sections