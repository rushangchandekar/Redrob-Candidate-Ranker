"""
Live Ranking Phase (Phase 2)
Orchestrates:
1. Loading precomputed assets (features, embeddings, FAISS index).
2. Querying FAISS index for top-K candidates.
3. Multi-dimensional scoring (5 dimensions) for retrieved candidates.
4. Honeypot, keyword stuffer, and consulting carrier trap detection.
5. Final ranking, twin deduplication, and top-100 selection.
6. Generating detailed 1-2 sentence recruiter reasoning.
7. Writing final submission.csv.

Must run in under 5 minutes on CPU.
"""

import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# Add workspace root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.faiss_index import load_faiss_index, search_top_k
from src.scoring.semantic_scorer import compute_semantic_scores
from src.scoring.career_scorer import score_career_trajectory
from src.scoring.skills_scorer import score_skills_depth
from src.scoring.behavioral_scorer import score_behavioral_signals
from src.scoring.activity_scorer import score_platform_activity
from src.detection.trap_detector import detect_traps
from src.ranking.ranker import rank_candidates
from src.ranking.reasoning import generate_reasoning
from src.output.csv_builder import build_submission_csv

def run_ranking(output_csv: str = "submission.csv"):
    start_time = time.time()
    print("=" * 60)
    print("STARTING REDROB PHASE 2 RANKING PIPELINE")
    print("=" * 60)
    
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    jd_profile_path = os.path.join(data_dir, "jd_profile.json")
    jd_embeddings_path = os.path.join(data_dir, "jd_embeddings.pkl")
    features_path = os.path.join(data_dir, "features.parquet")
    embeddings_path = os.path.join(data_dir, "candidate_embeddings.npy")
    index_path = os.path.join(data_dir, "faiss.index")
    
    # Check if precomputed assets exist
    required_assets = [jd_profile_path, jd_embeddings_path, features_path, embeddings_path, index_path]
    for asset in required_assets:
        if not os.path.exists(asset):
            print(f"Error: Required precomputed asset not found: {asset}")
            print("Please run scripts/run_preprocessing.py first.")
            sys.exit(1)
            
    # 2. Load Assets
    print("Loading precomputed assets...")
    load_start = time.time()
    
    with open(jd_profile_path, "r", encoding="utf-8") as f:
        jd_profile = pickle.load(f) if jd_profile_path.endswith(".pkl") else pd.read_json(jd_profile_path, typ="series").to_dict()
        
    with open(jd_embeddings_path, "rb") as f:
        jd_embeddings = pickle.load(f)
        
    df_features = pd.read_parquet(features_path)
    # Convert dataframe to list of dicts for faster iteration
    candidate_features = df_features.to_dict(orient="records")
    
    candidate_embeddings = np.load(embeddings_path).astype("float32")
    index = load_faiss_index(index_path)
    
    print(f"Loaded assets in {time.time() - load_start:.2f} seconds.")
    print(f"Total candidates in database: {len(candidate_features)}")
    
    # 3. Retrieve Top-K candidates using FAISS
    # We retrieve the top 1000 candidates for detailed scoring.
    # If the database has fewer than 1000 (e.g. sample mode), retrieve all.
    k_retrieve = min(1000, len(candidate_features))
    print(f"Querying FAISS index for top-{k_retrieve} candidates...")
    search_start = time.time()
    
    query_vector = jd_embeddings["mega"]
    faiss_scores, faiss_indices = search_top_k(index, query_vector, k=k_retrieve)
    
    print(f"FAISS search completed in {time.time() - search_start:.4f} seconds.")
    
    # 4. Multi-Dimensional Scoring
    print("Scoring candidates across 5 dimensions...")
    scoring_start = time.time()
    
    # D1: Semantic Match Score
    # Fetch embeddings of retrieved candidates
    subset_embeddings = candidate_embeddings[faiss_indices]
    semantic_scores = compute_semantic_scores(subset_embeddings, jd_embeddings)
    
    # D2-D5: Loop through retrieved candidates
    career_scores = []
    skills_scores = []
    behavioral_scores = []
    activity_scores = []
    trap_penalties = []
    candidate_traps = []
    
    for i, idx in enumerate(faiss_indices):
        cand_feats = candidate_features[idx]
        
        # D2: Career Trajectory
        career_score = score_career_trajectory(cand_feats, jd_profile)
        career_scores.append(career_score)
        
        # D3: Skills Depth
        skills_score = score_skills_depth(cand_feats, jd_profile)
        skills_scores.append(skills_score)
        
        # D4: Behavioral Signals (returns score and violations)
        beh_score, violations = score_behavioral_signals(cand_feats, jd_profile)
        behavioral_scores.append(beh_score)
        
        # D5: Platform Activity
        act_score = score_platform_activity(cand_feats)
        activity_scores.append(act_score)
        
        # Trap detection
        penalty, reasons = detect_traps(cand_feats, violations)
        trap_penalties.append(penalty)
        candidate_traps.append(reasons)
        
    dimension_scores = {
        "semantic": semantic_scores,
        "career": np.array(career_scores),
        "skills": np.array(skills_scores),
        "behavioral": np.array(behavioral_scores),
        "activity": np.array(activity_scores)
    }
    
    print(f"Multi-dimensional scoring completed in {time.time() - scoring_start:.2f} seconds.")
    
    # 5. Final Ranking and Twin Deduplication
    print("Ranking and deduplicating behavioral twins...")
    ranking_start = time.time()
    
    weights = {
        "semantic": 0.25,
        "career": 0.25,
        "skills": 0.20,
        "behavioral": 0.20,
        "activity": 0.10
    }
    
    top_results = rank_candidates(
        top_k_indices=faiss_indices,
        dimension_scores=dimension_scores,
        trap_penalties=np.array(trap_penalties),
        candidate_embeddings=candidate_embeddings,
        weights=weights,
        twin_threshold=0.98
    )
    
    print(f"Ranking and deduplication completed in {time.time() - ranking_start:.2f} seconds.")
    print(f"Deduplicated shortlist count: {len(top_results)}")
    
    # Take top 100
    top_100 = top_results[:100]
    
    # 6. Generate Reasoning for Top-100 Candidates
    print("Generating recruiter reasoning for top-100 shortlist...")
    reasoning_start = time.time()
    
    for result in tqdm(top_100, desc="Reasoning"):
        idx = result["original_index"]
        cand_feats = candidate_features[idx]
        # Generate reasoning
        reasoning = generate_reasoning(cand_feats, result)
        result["reasoning"] = reasoning
        
    print(f"Reasoning generated in {time.time() - reasoning_start:.2f} seconds.")
    
    # 7. Write to CSV
    output_path = os.path.join(base_dir, output_csv)
    print(f"Building final submission CSV at {output_path}...")
    build_submission_csv(top_100, candidate_features, output_path)
    
    # 8. Report honeypots self-audit
    honeypot_count = 0
    for res in top_100:
        idx = res["original_index"]
        cand_feats = candidate_features[idx]
        # Recalculate violations to count honeypot flags
        _, violations = score_behavioral_signals(cand_feats, jd_profile)
        penalty, reasons = detect_traps(cand_feats, violations)
        if "impossible_timeline" in reasons or any("signal_violations" in r for r in reasons):
            honeypot_count += 1
            
    print(f"Self-audit: Estimated honeypots in top-100: {honeypot_count} (Allowed: < 10)")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"RANKING COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print(f"Output saved to {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Redrob Hackathon Phase 2 Ranking")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", help="Path to candidates dataset")
    parser.add_argument("--out", type=str, default="submission.csv", help="Output CSV path (default: submission.csv)")
    args = parser.parse_args()
    
    run_ranking(output_csv=args.out)
