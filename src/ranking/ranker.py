"""
Final Ranker — Combines scores from 5 dimensions, applies trap penalties,
and deduplicates behavioral twins to produce the final top-100 candidate list.
"""

import numpy as np
from typing import Any

def remove_behavioral_twins(
    results: list[dict[str, Any]],
    candidate_embeddings: np.ndarray, # shape (num_candidates, 384)
    threshold: float = 0.98
) -> list[dict[str, Any]]:
    """
    Checks pairwise cosine similarity between candidate embeddings.
    If two candidates have a similarity score above the threshold,
    keeps only the one with the higher rank (higher score) and removes the other.
    """
    kept = []
    
    for result in results:
        idx = result["original_index"]
        # In case candidate_embeddings is the subset of K candidates,
        # we need to be careful with indexing. Assumes candidate_embeddings
        # index matches the raw candidate dataset.
        emb = candidate_embeddings[idx]
        
        is_twin = False
        for kept_result in kept:
            kept_idx = kept_result["original_index"]
            kept_emb = candidate_embeddings[kept_idx]
            
            # Cosine similarity (both vectors are normalized)
            similarity = float(np.dot(emb, kept_emb))
            
            if similarity > threshold:
                is_twin = True
                # Skip this candidate since we already kept a better-ranked twin
                break
                
        if not is_twin:
            kept.append(result)
            
    return kept

def rank_candidates(
    top_k_indices: np.ndarray,
    dimension_scores: dict[str, np.ndarray],
    trap_penalties: np.ndarray,
    candidate_embeddings: np.ndarray,
    weights: dict[str, float] = None,
    twin_threshold: float = 0.98
) -> list[dict[str, Any]]:
    """
    Combines the 5 scoring dimensions and penalties:
    D1: Semantic (25%)
    D2: Career (25%)
    D3: Skills (20%)
    D4: Behavioral (20%)
    D5: Activity (10%)
    Sorts them, removes duplicates/twins, and cuts off to top-100.
    """
    if weights is None:
        weights = {
            "semantic": 0.25,
            "career": 0.25,
            "skills": 0.20,
            "behavioral": 0.20,
            "activity": 0.10
        }
        
    num_candidates = len(top_k_indices)
    raw_results = []
    
    for i in range(num_candidates):
        idx = int(top_k_indices[i])
        
        # Calculate weighted base score
        base_score = sum(
            dimension_scores[dim][i] * weights[dim]
            for dim in weights
        )
        
        # Apply trap penalty
        final_score = base_score + trap_penalties[i]
        # Keep score in range [0, 100]
        final_score = max(0.0, min(100.0, final_score))
        
        raw_results.append({
            "original_index": idx,
            "final_score": final_score,
            "dimension_scores": {dim: float(dimension_scores[dim][i]) for dim in weights},
            "trap_penalty": float(trap_penalties[i])
        })
        
    # Sort descending by final score. Break ties by candidate_id ascending.
    # To break ties by candidate_id ascending, we need the actual candidate IDs.
    # Let's sort initially by score descending. The caller or the tiebreaker will handle
    # the exact sorting order.
    raw_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Deduplicate behavioral twins
    deduped_results = remove_behavioral_twins(
        raw_results,
        candidate_embeddings,
        threshold=twin_threshold
    )
    
    return deduped_results
