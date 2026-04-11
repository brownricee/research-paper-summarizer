import spacy
from spacy.util import is_package
import subprocess
import sys

from src.config import SENTENCES_PER_CHUNK

_nlp = None

def get_nlp():
  global _nlp
  if _nlp is None:
    try:
      _nlp = spacy.load("en_core_web_sm")
    except OSError:
      print("Downloading spaCy model 'en_core_web_sm'...")
      subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
      _nlp = spacy.load("en_core_web_sm")
  return _nlp


def chunk_text(sections, sentences_per_chunk=SENTENCES_PER_CHUNK):
  nlp = get_nlp()

  all_chunks = []

  for section, content in sections.items():
     doc = nlp(text=content)
     sentences = [sent.text for sent in doc.sents]

     chunks = []

     for i in range(0, len(sentences), sentences_per_chunk):
      # slices the sentences so we have 5 per chunk
      chunk = " ".join(sentences[i:i + sentences_per_chunk])
      chunks.append({"section": section, "text": chunk})

     all_chunks.extend(chunks)

  return all_chunks
