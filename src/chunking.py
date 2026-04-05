import spacy
from spacy.util import is_package
import subprocess
import sys


def chunk_text(sections, sentences_per_chunk=5):
  # Ensure en_core_web_sm is installed
  try:
    nlp = spacy.load("en_core_web_sm")
  except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


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
