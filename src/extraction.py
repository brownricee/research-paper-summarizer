import pdfplumber
from src.preprocessing import clean_text

def getTextFromPDF(pdf_path):
  text = ""
  try:
    with pdfplumber.open(pdf_path) as pdf:
      for page in pdf.pages:
        page_text = page.extract_text()
        text += page_text
  except FileNotFoundError:
    print("The file you tried to provide does not exist.")

  cleaned_text = clean_text(text)

  return cleaned_text