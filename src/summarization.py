from transformers import pipeline
from transformers import AutoTokenizer


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained(summarizer.tokenizer)
