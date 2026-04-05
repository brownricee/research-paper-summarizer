from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def rank_chunks(chunks):
    chunk_embeddings = model.encode(chunks)
    
    # compute centroid
    # score each chunk by cosine similarity to centroid
    # return chunks sorted by score!
    pass
