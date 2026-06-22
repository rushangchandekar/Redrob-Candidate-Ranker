"""
Semantic Scorer — Computes candidate semantic matching scores based on JD requirements.
"""

import numpy as np
from typing import Any

def compute_semantic_scores(
    candidate_embeddings: np.ndarray,   # shape (K, 384) for candidate shortlist
    jd_embeddings: dict[str, Any]       # loaded from jd_embeddings.pkl
) -> np.ndarray:
    """
    Computes a weighted average cosine similarity across all individual JD requirements.
    Returns scores scaled to 0-100, shape (K,).
    """
    req_embeddings = jd_embeddings["requirements"]   # shape (num_req, 384)
    weights = np.array(jd_embeddings["weights"])     # shape (num_req,)
    
    # Calculate similarity: matrix product of normalized embeddings (cosine similarity)
    # similarity[i, j] is similarity of candidate i with requirement j
    similarity = candidate_embeddings @ req_embeddings.T   # shape (K, num_req)
    
    # Clip to [0, 1] to avoid negative or slightly > 1.0 values
    similarity = np.clip(similarity, 0.0, 1.0)
    
    # Weighted average similarity per candidate
    weighted_scores = (similarity @ weights) / np.sum(weights)  # shape (K,)
    
    # Scale to 0-100
    return weighted_scores * 100.0
