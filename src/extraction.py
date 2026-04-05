import pdfplumber
from src.preprocessing import clean_text
from src.preprocessing import detect_sections

def getTextFromPDF(pdf_path):
  text = ""
  try:
    with pdfplumber.open(pdf_path) as pdf:
      for page in pdf.pages:
        page_text = page.extract_text()
        text += page_text
  except FileNotFoundError:
    print("The file you tried to provide does not exist.")

  sections = detect_sections(text)

  for key, value in sections.items():
    value = clean_text(value)
    sections[key] = value

  return sections