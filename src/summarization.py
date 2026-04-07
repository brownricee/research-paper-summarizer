from transformers import pipeline
from transformers import AutoTokenizer


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
