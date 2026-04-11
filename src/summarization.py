from transformers import pipeline
from collections import defaultdict

from src.config import (
    SUMMARIZER_MODEL,
    TOP_N_CHUNKS,
    MIN_CHUNK_CHARS,
    SUMMARY_MAX_LENGTH,
    SUMMARY_MIN_LENGTH,
    TOP_N_PER_SECTION,
)

_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
    return _summarizer

def summarize_chunks(ranked_chunks, top_n=TOP_N_PER_SECTION):
    section_summaries = {}
    summarizer = get_summarizer()

    buckets = defaultdict(list)
    for chunk in ranked_chunks:
        buckets[chunk["section"]].append(chunk)

    for section_name, chunks_in_section in buckets.items():
        seen_texts = []
        section_summary = ""

        for i in range(min(top_n, len(chunks_in_section))):
            text = chunks_in_section[i]["text"]

            # Skip chunks that are too short to summarize meaningfully
            if len(text) < MIN_CHUNK_CHARS:
                continue

            # Skip chunks that are too similar to already-summarized ones in this section
            if any(text in seen or seen in text for seen in seen_texts):
                continue
            seen_texts.append(text)

            # Cap max_length to half the input token count so we don't exceed input length
            input_tokens = len(text.split())
            max_len = min(SUMMARY_MAX_LENGTH, max(SUMMARY_MIN_LENGTH, input_tokens // 2))

            result = summarizer(text, max_length=max_len, min_length=min(20, max_len - 1), do_sample=False)
            section_summary += result[0]["summary_text"]
            section_summary += "\n"

        section_summaries[section_name] = section_summary

    return section_summaries