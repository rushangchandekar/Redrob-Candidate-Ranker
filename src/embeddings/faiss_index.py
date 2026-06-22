"""
FAISS Index Builder & Searcher — Handles FAISS Index Flat Inner Product construction
and nearest neighbor search for high-speed candidate retrieval.
"""

import os
import faiss
import numpy as np


def build_faiss_index(embeddings_path: str, index_path: str) -> faiss.IndexFlatIP:
    """
    Load precomputed candidate embeddings, build FAISS flat inner-product index,
    and save index to disk. L2-normalized vectors mean Inner Product = Cosine Similarity.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
        
    embeddings = np.load(embeddings_path).astype("float32")
    dim = embeddings.shape[1]
    
    print(f"Building FAISS IndexFlatIP with dimension {dim} for {embeddings.shape[0]} candidates...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index written successfully to {index_path}")
    return index

def load_faiss_index(index_path: str) -> faiss.IndexFlatIP:
    """Load an existing FAISS index from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return faiss.read_index(index_path)

def search_top_k(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    k: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the index for the top-k nearest neighbors of the query vector.
    Returns (scores, indices) arrays of shape (k,).
    """
    # Ensure query embedding is 2D and float32
    query = query_embedding.reshape(1, -1).astype("float32")
    
    # Search index
    scores, indices = index.search(query, k)
    
    # Return 1D arrays
    return scores[0], indices[0]
