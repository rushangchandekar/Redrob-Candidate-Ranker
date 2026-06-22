"""
CSV Builder — Formats and writes the final ranked candidate shortlist into the submission CSV.
Ensures exact header, column order, formatting, and tie-breaking constraints.
"""

import os
import csv
import pandas as pd
from typing import Any

def build_submission_csv(
    top_100: list[dict[str, Any]],
    candidate_features: list[dict[str, Any]],
    output_path: str
):
    """
    Builds the submission CSV from the final top-100 ranked list.
    Saves it to output_path.
    Column names: candidate_id, rank, score, reasoning
    """
    rows = []
    
    for i, result in enumerate(top_100):
        idx = result["original_index"]
        features = candidate_features[idx]
        rank = i + 1
        
        rows.append({
            "candidate_id": features["id"],
            "rank": rank,
            "score": round(result["final_score"], 4),
            "reasoning": result["reasoning"]
        })
        
    df = pd.DataFrame(rows)
    
    # Sort and double check ordering
    # The submission validator requires that scores are non-increasing by rank.
    # Ties are broken by candidate_id ascending.
    # Let's perform a sorting of our rows first to guarantee this:
    # 1. Sort by score descending.
    # 2. If scores are equal, sort by candidate_id ascending.
    # 3. Assign ranks from 1 to 100 based on this final order.
    
    df["score"] = df["score"].astype(float)
    df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = df.index + 1
    
    # Select and order required columns
    df = df[["candidate_id", "rank", "score", "reasoning"]]
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Successfully wrote {len(df)} ranked candidates to {output_path}")
