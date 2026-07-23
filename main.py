import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.extraction import getTextFromPDF
from src.chunking import chunk_text
from src.ranking import rank_chunks
from src.summarization import summarize_chunks
from src.aggregation import aggregate_summary
from src.summarization import synthesize_summary

def main():
    pdf_path = "C:\\Users\\ryaan\\Desktop\\research-paper-summarizer\\attention_research_paper.pdf"

    sections = getTextFromPDF(pdf_path)

    chunks = chunk_text(sections)

    ranked_chunks = rank_chunks(chunks)

    chunk_summaries = summarize_chunks(ranked_chunks)

    combined_summary = aggregate_summary(chunk_summaries, section_order=list(sections.keys()))

    final_summary = synthesize_summary(combined_summary)

    print(final_summary)



if __name__ == "__main__":
    main()