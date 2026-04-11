from transformers import pipeline

from src.config import (
    SUMMARIZER_MODEL,
    TOP_N_CHUNKS,
    MIN_CHUNK_CHARS,
    SUMMARY_MAX_LENGTH,
    SUMMARY_MIN_LENGTH,
)

summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)

def summarize_chunks(ranked_chunks, top_n=TOP_N_CHUNKS):
    summary = ""
    seen_texts = []

    for i in range(min(top_n, len(ranked_chunks))):
        text = ranked_chunks[i]["text"]

        # Skip chunks that are too short to summarize meaningfully
        if len(text) < MIN_CHUNK_CHARS:
            continue

        # Skip chunks that are too similar to already-summarized ones
        if any(text in seen or seen in text for seen in seen_texts):
            continue
        seen_texts.append(text)

        # Cap max_length to half the input token count so we don't exceed input length
        input_tokens = len(text.split())
        max_len = min(SUMMARY_MAX_LENGTH, max(SUMMARY_MIN_LENGTH, input_tokens // 2))

        result = summarizer(text, max_length=max_len, min_length=min(20, max_len - 1), do_sample=False)
        summary += result[0]["summary_text"]
        summary += "\n"

    return summary

