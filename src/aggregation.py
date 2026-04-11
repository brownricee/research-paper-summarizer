from src.summarization import summarizer

def aggregate_summary(chunk_summaries):
    if chunk_summaries:
        chunk_summaries = summarizer(chunk_summaries, max_length=300, min_length=100, do_sample=False)
        summary_text = chunk_summaries[0]["summary_text"]
        return summary_text
    else:
        return ""