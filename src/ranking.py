from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

from src.config import EMBEDDING_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def rank_chunks(chunks):
    model = get_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    
    grouped_sections = defaultdict(list)

    for i, chunk in enumerate(chunks):
        grouped_sections[chunk["section"]].append(i)

    for section in grouped_sections:
        indices = grouped_sections[section]
        rows = embeddings[indices]

        # Compute local centroid (which chunks for each section represent their overall meaning)
        local_centroid = np.mean(rows, axis=0)
        # Score text by its cosine similarity to centroid
        scores = cosine_similarity(rows, local_centroid.reshape(1, -1))
        # Write scores back to chunk
        for chunk_index, score in zip(indices, scores.flatten()):
            chunks[chunk_index]["score"] = score

    return chunks