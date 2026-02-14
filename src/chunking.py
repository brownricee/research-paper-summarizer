import pdfplumber
import spacy
import re

def chunk_text(text, sentences_per_chunk=5):

  nlp = spacy.load("en_core_web_sm")
  doc = nlp(text)
  sentences = [sent.text for sent in doc.sents]
  chunks = []

  for i in range(0, len(sentences), sentences_per_chunk):
    # slices the sentences so we have 5 per chunk
    chunk = "".join(sentences[i:i + sentences_per_chunk])
    chunks.append(chunk)

  return chunks
