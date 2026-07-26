from transformers import pipeline
from collections import defaultdict

from src.chunking import get_nlp

from src.config import (
    SUMMARIZER_MODEL,
    MIN_CHUNK_CHARS,
    SUMMARY_MAX_LENGTH,
    SUMMARY_MIN_LENGTH,
    TOP_N_PER_SECTION,
    AGGREGATE_MAX_LENGTH,
    AGGREGATE_MIN_LENGTH,
)

_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
        _summarizer.tokenizer.model_max_length = 1024
    return _summarizer

def _clamp_lengths(text, max_cap, min_cap):
    # Use the real input length so the output never asks for more
    # tokens than the input has, and min_length never exceeds max_length.
    tokenizer = get_summarizer().tokenizer
    input_len = len(tokenizer.encode(text, truncation=True, max_length=1024, add_special_tokens=False))
    max_len = min(max_cap, input_len)
    min_len = min(min_cap, max_len)
    return max_len, min_len

def summarize_chunks(ranked_chunks, top_n=TOP_N_PER_SECTION):
    section_summaries = {}
    summarizer = get_summarizer()

    buckets = defaultdict(list)
    for chunk in ranked_chunks:
        buckets[chunk["section"]].append(chunk)

    for section_name, chunks_in_section in buckets.items():
        seen_texts = []
        section_summary = ""

        chunks_in_section.sort(key=lambda chunk: chunk["score"], reverse=True)

        for i in range(min(top_n, len(chunks_in_section))):
            text = chunks_in_section[i]["text"]

            # Skip chunks that are too short to summarize meaningfully
            if len(text) < MIN_CHUNK_CHARS:
                continue

            # Skip chunks that are too similar to already-summarized ones in this section
            if any(text in seen or seen in text for seen in seen_texts):
                continue
            seen_texts.append(text)

            max_len, min_len = _clamp_lengths(text, SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH)

            result = summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                no_repeat_ngram_size=3,
                truncation=True,
                num_beams=4,
            )
            section_summary += result[0]["summary_text"]
            section_summary += "\n"

        section_summaries[section_name] = section_summary

    return section_summaries


def synthesize_summary(combined_text):
    summarizer = get_summarizer()
    combined_text = reduce_to_fit(text=combined_text)
    max_len, min_len = _clamp_lengths(combined_text, AGGREGATE_MAX_LENGTH, AGGREGATE_MIN_LENGTH)
    result = summarizer(
        combined_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        no_repeat_ngram_size=3,
        truncation=True,
        num_beams=4,
    )

    return result[0]["summary_text"]

def token_len(text):
    summarizer = get_summarizer()
    tokenizer = summarizer.tokenizer
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_by_tokens(text, limit=900):
    nlp = get_nlp()
    doc = nlp(text)
    windows = []
    current = []
    current_tokens = 0

    sentences = [sent.text for sent in doc.sents]

    for sentence in sentences:
        sent_tokens = token_len(sentence)
        if current and current_tokens + sent_tokens > limit:
            # Turns accumulated sentences into one string and makes it its own window.
            windows.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sent_tokens

    # Any incomplete windows are added onto the end before we finish.
    if current:
        windows.append(" ".join(current))

    return windows

def reduce_to_fit(text, limit=1000):
    max_iteration = 5
    iterations = 0
    summarizer = get_summarizer()

    while token_len(text) > limit and iterations < max_iteration:
        windows = split_by_tokens(text, 900)
        summaries = []
        for w in windows:
            max_len, min_len = _clamp_lengths(w, SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH)
            result = summarizer(w,
                                max_length=max_len,
                                min_length=min_len,
                                do_sample=False,
                                no_repeat_ngram_size=3,
                                truncation=True,
                                num_beams=4)
            summaries.append(result[0]["summary_text"])
        text = "\n".join(summaries)

        iterations += 1
    # combined text now summarized properly without discarding information early due to 1024 token truncation
    return text
