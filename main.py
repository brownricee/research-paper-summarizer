import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.extraction import getTextFromPDF
from src.chunking import chunk_text
from src.ranking import rank_chunks
from src.summarization import summarize_chunks
from src.aggregation import aggregate_summary
from src.summarization import synthesize_summary
from src.config import (
    TLDR_MAX_LENGTH,
    TLDR_MIN_LENGTH,
    TLDR_SECTIONS
)

def main():
    pdf_path = "C:\\Users\\ryaan\\Desktop\\research-paper-summarizer\\attention_research_paper.pdf"

    sections = getTextFromPDF(pdf_path)
    print("Text extracted..\n")

    chunks = chunk_text(sections)
    print("Chunking text..\n")

    ranked_chunks = rank_chunks(chunks)
    print("Ranking chunks..\n")

    chunk_summaries = summarize_chunks(ranked_chunks)
    print("Summarizing..\n")

    key_list = [k for k in sections if k != 'abstract']
    body = aggregate_summary(chunk_summaries, section_order=key_list, with_headers=True)
    plain = aggregate_summary(chunk_summaries, section_order=list(sections.keys()), with_headers=False)

    tldr_keys = [k for k in sections if any(s in k for s in TLDR_SECTIONS)]
    tldr_source = aggregate_summary(chunk_summaries, section_order=tldr_keys, with_headers=False) or plain

    tldr = synthesize_summary(tldr_source, TLDR_MAX_LENGTH, TLDR_MIN_LENGTH) if tldr_source else ""

    title = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").title()
    document = f"# Summary: {title}\n\n## TL;DR\n{tldr}\n\n{body}"

    print(document)



if __name__ == "__main__":
    main()