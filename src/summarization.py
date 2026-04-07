from transformers import pipeline


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_chunks(ranked_chunks, top_n=8):
    summary = ""
    for i in range(min(top_n, len(ranked_chunks))):
        result = summarizer(ranked_chunks[i]["text"], max_length=150, min_length=40, do_sample=False)
        summary += result[0]["summary_text"]
        summary += "\n"

    return summary

