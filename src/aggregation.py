from src.summarization import get_summarizer
from src.config import AGGREGATE_MAX_LENGTH, AGGREGATE_MIN_LENGTH

def aggregate_summary(chunk_summaries):
    if chunk_summaries:
        summarizer = get_summarizer()
        chunk_summaries = summarizer(chunk_summaries, max_length=AGGREGATE_MAX_LENGTH, min_length=AGGREGATE_MIN_LENGTH, do_sample=False)
        summary_text = chunk_summaries[0]["summary_text"]
        return summary_text
    else:
        return ""