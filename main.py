import torch
import numpy as np
import matplotlib as plt

from src.preprocessing import clean_text
from src.extraction import getTextFromPDF
from src.chunking import chunk_text

def main():
    pdf_path = "C:\\Users\\ryaan\\OneDrive\\Desktop\\paper-summarizer\\example_research_paper.pdf"

    sections = getTextFromPDF(pdf_path)

    chunks = chunk_text(sections)



if __name__ == "__main__":
    main()
