from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def rank_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    # Compute centroid (which chunks best represent overall meaning of paper)
    centroid = np.mean(embeddings, axis=0)
    # score each chunk by cosine similarity to centroid
    scores = cosine_similarity(embeddings, centroid.reshape(1, -1))
    # sort in ascending order and flatten so its a 1-dimensional array
    sorted_indices = np.argsort(scores.flatten(), axis=None)[::-1]

    # copies previous dictionary and continues adding to it (now with scores)
    return [{**chunks[i], "score": scores.flatten()[i]} for i in sorted_indices]
