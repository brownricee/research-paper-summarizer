import re
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
    NUM_BEAMS,
    COMPRESSION_RATIO
)

_summarizer = None

# Fixes issue with punctuation appearing after a space due to distillbart model.
def _normalize_spacing(text):
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"[ \t]+", " ", text).strip()

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
    target_len = max(min_cap, int(input_len * COMPRESSION_RATIO))
    max_len = min(max_cap, target_len)
    min_len = min(min_cap, max_len)
    return max_len, min_len

def summarize_chunks(ranked_chunks, top_n=TOP_N_PER_SECTION):
    section_summaries = {}

    buckets = defaultdict(list)

    for position, chunk in enumerate(ranked_chunks):
        buckets[chunk["section"]].append((position, chunk))

    for section_name, chunks_in_section in buckets.items():
        seen_texts = []
        kept_texts = []

        chunks_in_section.sort(key=lambda pair: pair[1]["score"], reverse=True)
        selected = chunks_in_section[:top_n]
        selected.sort(key=lambda pair: pair[0])

        for position, chunk in selected:
            text = chunk["text"]

            # Skip chunks that are too short to summarize meaningfully
            if len(text) < MIN_CHUNK_CHARS:
                continue

            # Skip chunks that are too similar to already-summarized ones in this section
            if any(text in seen or seen in text for seen in seen_texts):
                continue
            seen_texts.append(text)

            kept_texts.append(text)

        if not kept_texts:
            section_summaries[section_name] = ""
            continue

        section_input = " ".join(kept_texts)
        section_input = reduce_to_fit(section_input)

        section_summaries[section_name] = _run_summarizer(
            section_input, SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH
        )

    return section_summaries


def synthesize_summary(combined_text, max_cap=AGGREGATE_MAX_LENGTH, min_cap=AGGREGATE_MIN_LENGTH):
    combined_text = reduce_to_fit(text=combined_text)

    return _run_summarizer(combined_text, max_cap, min_cap)

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

    while token_len(text) > limit and iterations < max_iteration:
        windows = split_by_tokens(text, 900)
        summaries = []
        for w in windows:
            summaries.append(_run_summarizer(w, SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH))
        text = "\n".join(summaries)

        iterations += 1
    # Combined text now summarized properly without discarding information early due to 1024 token truncation
    return text

def _run_summarizer(text, max_cap, min_cap, allow_empty=False):
    summarizer = get_summarizer()
    max_len, min_len = _clamp_lengths(text, max_cap, min_cap)
    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        no_repeat_ngram_size=3,
        truncation=True,
        num_beams=NUM_BEAMS,
    )

    result = _normalize_spacing(result[0]["summary_text"])

    result = _drop_dangling_sentences(result)

    return _trim_to_last_sentence(result, allow_empty=allow_empty)

def _trim_to_last_sentence(text, allow_empty=False):
    text = text.strip()
    trailing_closer = ("\"", "'", ")", "]", "}", "\u201d", "\u2019")
    terminator_characters = (".", "?", "!")

    def ends_sentence(candidate):
        # Peel trailing closers and whitespace so '...all you need."' still counts.
        candidate = candidate.rstrip("".join(trailing_closer) + " \t\r\n")
        if not candidate:
            return False
        return candidate.endswith(terminator_characters)

    if not text:
        return text
    elif ends_sentence(text):
        return text

    nlp = get_nlp()
    doc = nlp(text)

    spans = list(doc.sents)

    for i in range(len(spans) - 1, -1, -1):
        sentence = spans[i].text
        if ends_sentence(sentence):
            return text[:spans[i].end_char]

    # No complete sentence anywhere: never return empty.
    return "" if allow_empty else text

DANGLING_TAIL_RE = re.compile(
    r"\b(?:such as|similar to|available at|described in|shown in"
    r"|proposed in|introduced in|of|in|at)\s*[.!?]$",
    re.IGNORECASE,
)

def _drop_dangling_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = [s for s in sentences if not DANGLING_TAIL_RE.search(s.strip())]
    if not kept:
        return text
    return " ".join(kept)