"""
Candidate Embedder — Generates embeddings for candidate profiles using sentence-transformers.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union

def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence transformer model.
    Downloads from Hugging Face if not cached, otherwise loads from local cache.
    Using 'all-MiniLM-L6-v2' (384-dimensional, lightweight, strong NLP/IR performance).
    """
    # Force CPU usage as per constraints
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return model

def embed_texts(texts: list[str], batch_size: int = 512, show_progress: bool = True) -> np.ndarray:
    """
    Generate L2-normalized embeddings for a list of text strings.
    Normalization is critical so that Inner Product = Cosine Similarity in FAISS.
    """
    model = load_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.astype("float32")

def embed_jd_profile(jd_profile: dict, output_path: str):
    """
    Embed the job description:
    1. The 'jd_mega_text' (for FAISS retrieval)
    2. Each hard/soft requirement separately (for granular multi-dimensional scoring)
    """
    model = load_embedding_model()
    
    # 1. Embed mega text
    mega_text = jd_profile["jd_mega_text"]
    mega_embedding = model.encode([mega_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    
    # 2. Embed requirements
    hard_reqs = jd_profile.get("hard_requirements", [])
    soft_reqs = jd_profile.get("soft_requirements", [])
    all_reqs = hard_reqs + soft_reqs
    
    req_texts = [r["requirement"] for r in all_reqs]
    req_weights = [r["weight"] for r in all_reqs]
    req_keys = [r.get("key", "unknown") for r in all_reqs]
    req_types = [r.get("type", "skill") for r in all_reqs]
    
    req_embeddings = model.encode(req_texts, normalize_embeddings=True, convert_to_numpy=True)
    
    data_to_save = {
        "mega": mega_embedding,
        "requirements": req_embeddings,
        "weights": req_weights,
        "texts": req_texts,
        "keys": req_keys,
        "types": req_types
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Saved JD embeddings to {output_path}")

def embed_candidates_from_file(texts_path: str, output_path: str, batch_size: int = 512):
    """
    Load pre-built candidate text representation lists and generate full embeddings.
    Used during offline pre-computation.
    """
    if not os.path.exists(texts_path):
        raise FileNotFoundError(f"Candidate texts file not found at {texts_path}")
        
    with open(texts_path, "rb") as f:
        candidate_texts = pickle.load(f)
        
    print(f"Embedding {len(candidate_texts)} candidates...")
    embeddings = embed_texts(candidate_texts, batch_size=batch_size, show_progress=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved candidate embeddings of shape {embeddings.shape} to {output_path}")
