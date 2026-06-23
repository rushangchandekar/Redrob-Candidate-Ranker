"""
Offline Pre-computation Pipeline (Phase 1)
Orchestrates:
1. JD analysis and JD embedding.
2. Ingestion and preprocessing of candidate profiles.
3. Creation of text corpus for embeddings.
4. Candidate embedding generation (runs sentence-transformers).
5. FAISS index building.

Usage:
    python scripts/run_preprocessing.py --sample   # Runs on sample data (50 candidates)
    python scripts/run_preprocessing.py --full     # Runs on full dataset (100,000 candidates)
"""

import os
import sys
import argparse
import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add workspace root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.loader import detect_dataset_path, load_all_candidates, load_sample_candidates, stream_candidates
from src.pipeline.preprocessor import preprocess_candidate
from src.pipeline.text_builder import build_candidate_text
from src.jd.analyzer import analyze_jd, save_jd_profile
from src.embeddings.embedder import embed_jd_profile, embed_texts
from src.embeddings.faiss_index import build_faiss_index

def run_pipeline(is_sample: bool = True, custom_candidates_path: str = None):
    start_time = time.time()
    print("=" * 60)
    print("STARTING REDROB PHASE 1 PIPELINE")
    print("=" * 60)
    
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    jd_path = os.path.join(base_dir, "job_description.docx")
    if not os.path.exists(jd_path):
        jd_path = os.path.join(base_dir, "job_description.md")
        
    jd_profile_path = os.path.join(data_dir, "jd_profile.json")
    jd_embeddings_path = os.path.join(data_dir, "jd_embeddings.pkl")
    
    features_path = os.path.join(data_dir, "features.parquet")
    texts_path = os.path.join(data_dir, "candidate_texts.pkl")
    embeddings_path = os.path.join(data_dir, "candidate_embeddings.npy")
    index_path = os.path.join(data_dir, "faiss.index")
    
    # 2. Parse and Embed JD
    print("\n--- Step 1: Analyzing Job Description ---")
    jd_profile = analyze_jd(jd_path)
    save_jd_profile(jd_profile, jd_profile_path)
    print("Embedding Job Description...")
    embed_jd_profile(jd_profile, jd_embeddings_path)
    
    # 3. Load Candidates
    print("\n--- Step 2: Loading Candidates ---")
    if custom_candidates_path:
        if custom_candidates_path.lower().endswith(".json"):
            candidates = load_sample_candidates(custom_candidates_path)
        else:
            candidates = []
            for cand in tqdm(stream_candidates(custom_candidates_path), desc="Ingesting profiles"):
                candidates.append(cand)
        print(f"Loaded {len(candidates)} candidates from {custom_candidates_path}")
    elif is_sample:
        sample_file = os.path.join(base_dir, "sample_candidates.json")
        candidates = load_sample_candidates(sample_file)
        print(f"Loaded {len(candidates)} candidates from {sample_file}")
    else:
        dataset_file = os.path.join(base_dir, "candidates.jsonl")
        if not os.path.exists(dataset_file):
            dataset_file = os.path.join(base_dir, "candidates.jsonl.gz")
        print(f"Streaming and preprocessing candidates from {dataset_file}...")
        # Since full jsonl can be large, we stream and preprocess in chunks
        candidates = []
        for cand in tqdm(stream_candidates(dataset_file), desc="Ingesting profiles"):
            candidates.append(cand)
        print(f"Loaded {len(candidates)} candidates.")
        
    # 4. Preprocess Candidates
    print("\n--- Step 3: Preprocessing Candidate Profiles ---")
    preprocessed_candidates = []
    candidate_texts = []
    
    for cand in tqdm(candidates, desc="Preprocessing"):
        features = preprocess_candidate(cand)
        text_repr = build_candidate_text(features)
        
        preprocessed_candidates.append(features)
        candidate_texts.append(text_repr)
        
    # Save preprocessed features to Parquet
    print(f"Saving preprocessed features to {features_path}...")
    df = pd.DataFrame(preprocessed_candidates)
    # Ensure lists and dicts are handled correctly (pandas parquet supports them)
    df.to_parquet(features_path, index=False)
    
    # Save candidate texts list
    print(f"Saving candidate texts corpus to {texts_path}...")
    with open(texts_path, "wb") as f:
        pickle.dump(candidate_texts, f)
        
    # 5. Embed Candidates
    print("\n--- Step 4: Generating Semantic Embeddings (Model: all-MiniLM-L6-v2) ---")
    embed_start = time.time()
    # Batch size is 512 for optimal CPU performance
    embeddings = embed_texts(candidate_texts, batch_size=512, show_progress=True)
    np.save(embeddings_path, embeddings)
    print(f"Generated embeddings of shape {embeddings.shape} in {time.time() - embed_start:.2f} seconds.")
    
    # 6. Build FAISS Index
    print("\n--- Step 5: Building FAISS Nearest Neighbor Index ---")
    build_faiss_index(embeddings_path, index_path)
    
    print("\n" + "=" * 60)
    print(f"PHASE 1 PIPELINE COMPLETE IN {time.time() - start_time:.2f} SECONDS!")
    print("All assets precomputed and saved in data/ directory.")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Redrob Hackathon Phase 1 Offline Pre-computation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="Run on 50 sample candidates")
    group.add_argument("--full", action="store_true", help="Run on full 100k candidate dataset")
    
    args = parser.parse_args()
    run_pipeline(is_sample=args.sample)
