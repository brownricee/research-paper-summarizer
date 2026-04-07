import torch
import numpy as np
import matplotlib as plt

from src.extraction import getTextFromPDF
from src.chunking import chunk_text
from src.ranking import rank_chunks
from src.summarization import summarize_chunks

def main():
    pdf_path = "C:\\Users\\ryaan\\OneDrive\\Desktop\\paper-summarizer\\example_research_paper.pdf"

    sections = getTextFromPDF(pdf_path)

    chunks = chunk_text(sections)

    ranked_chunks = rank_chunks(chunks)



if __name__ == "__main__":
    main()
