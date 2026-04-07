from transformers import pipeline


summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_chunks(ranked_chunks, top_n=5):
    summary = ""
    seen_texts = []

    for i in range(min(top_n, len(ranked_chunks))):
        text = ranked_chunks[i]["text"]

        # Skip chunks that are too short to summarize meaningfully
        if len(text) < 50:
            continue

        # Skip chunks that are too similar to already-summarized ones
        if any(text in seen or seen in text for seen in seen_texts):
            continue
        seen_texts.append(text)

        # Cap max_length to half the input token count so we don't exceed input length
        input_tokens = len(text.split())
        max_len = min(150, max(30, input_tokens // 2))

        result = summarizer(text, max_length=max_len, min_length=min(20, max_len - 1), do_sample=False)
        summary += result[0]["summary_text"]
        summary += "\n"

    return summary

