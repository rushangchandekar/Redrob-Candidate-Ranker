"""
Skills Scorer — Assesses quality, depth, and relevance of candidate skills.
Protects against keyword stuffing by penalizing large lists with shallow descriptions.
"""

from typing import Any

# Map skill groups to candidate normalized skills
HARD_SKILL_GROUPS = {
    "python": ["python", "py", "django", "flask", "fastapi"],
    "embeddings": [
        "embeddings", "sentence-transformers", "sentence transformers", "bge", "e5", 
        "retrieval", "semantic search", "dense retrieval", "nlp", "natural language processing"
    ],
    "vector_db": [
        "vector database", "vector search", "pinecone", "weaviate", "qdrant", "milvus", 
        "opensearch", "elasticsearch", "faiss", "hybrid search", "chromadb", "pgvector"
    ],
    "eval_frameworks": [
        "ndcg", "mrr", "map", "a/b testing", "ab testing", "evaluation", 
        "eval frameworks", "offline evaluation", "ranking", "recommender systems"
    ]
}

SOFT_SKILL_GROUPS = {
    "fine_tuning": ["fine-tuning", "fine tuning", "lora", "qlora", "peft", "sft", "rlhf"],
    "ltr": ["learning to rank", "ltr", "xgboost", "lightgbm", "neural ranking", "re-ranking"],
    "hr_tech": ["hr-tech", "recruiting tech", "marketplace", "talent intelligence"],
    "dist_systems": ["distributed systems", "inference optimization", "cuda", "onnx", "tensorrt", "spark", "triton"],
    "open_source": ["open-source", "open source", "github", "git"]
}

PROFICIENCY_MAP = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.9,
    "expert": 1.0
}

def get_skill_depth(skill_obj: dict) -> float:
    """Computes a depth score (0-1) for a specific skill based on proficiency, endorsements, and duration."""
    prof = skill_obj.get("proficiency", "beginner").lower()
    prof_weight = PROFICIENCY_MAP.get(prof, 0.3)
    
    endorsements = skill_obj.get("endorsements", 0)
    end_boost = min(0.2, endorsements / 50.0) # up to 0.2 boost for 50+ endorsements
    
    duration = skill_obj.get("duration_months", 0)
    dur_factor = min(1.0, duration / 36.0) # 3 years = full duration credit
    
    # Combined depth: 50% proficiency/endorsement, 50% duration
    depth = 0.5 * (prof_weight + end_boost) + 0.5 * dur_factor
    return min(1.0, depth)

def score_skills_depth(features: dict[str, Any], jd_profile: dict[str, Any] = None) -> float:
    """
    Evaluates candidate skills:
    - Hard skill match (70% weight): Checks the 4 critical categories
    - Soft skill match (30% weight): Checks the 5 preferred categories
    - Stuffing Penalty: Penalizes candidates listing too many skills but with shallow description depth.
    """
    # Raw skills are needed to access endorsements and proficiency
    # Wait, did we store raw skills or list of skill dicts?
    # In `preprocess_candidate`, we normalized the skills to list of strings.
    # To check proficiency and endorsements, we need the raw skill dictionaries!
    # Let's check: the raw candidate has the `skills` field as a list of dicts.
    # But wait, did we include `skills` in our preprocessed features?
    # In `preprocess_candidate` return: we had `skills_normalized`. We did not pass the full list of skill dicts.
    # Let's verify if we can pass the raw candidate's `skills` or modify preprocessor.py to include the raw skills dict list!
    # Wait, modifying `preprocessor.py` to include `"skills_raw": raw_candidate.get("skills", [])` is very easy and clean,
    # just like we did for `education`.
    # Let's modify `preprocessor.py` to include `"skills_raw": skills` so that we have full skill metadata during scoring!
    # Yes, let's do a replace_file_content on `preprocessor.py` first.
    
    raw_skills = features.get("skills_raw", [])
    if type(raw_skills).__name__ == 'ndarray':
        raw_skills = list(raw_skills)
    if raw_skills is None or len(raw_skills) == 0:
        return 0.0
        
    work_depth = features.get("work_description_depth", 0.5)
    skills_count = features.get("skills_count", 0)
        
    # Map normalized names to raw skill objects for quick lookup
    from src.pipeline.preprocessor import normalize_skill_name
    skills_map = {normalize_skill_name(s.get("name", "")): s for s in raw_skills if s.get("name")}
    
    # 1. Hard skill match (0-70 points)
    hard_scores = []
    for category, keywords in HARD_SKILL_GROUPS.items():
        cat_scores = []
        for kw in keywords:
            if kw in skills_map:
                cat_scores.append(get_skill_depth(skills_map[kw]))
        # Category score is the max depth found in that category
        hard_scores.append(max(cat_scores) if cat_scores else 0.0)
    
    # Hard score is the average across all 4 required categories
    hard_score = (sum(hard_scores) / 4.0) * 70.0
    
    # 2. Soft skill match (0-30 points)
    soft_scores = []
    for category, keywords in SOFT_SKILL_GROUPS.items():
        cat_scores = []
        for kw in keywords:
            if kw in skills_map:
                cat_scores.append(get_skill_depth(skills_map[kw]))
        soft_scores.append(max(cat_scores) if cat_scores else 0.0)
        
    # Soft score is the average across all 5 categories
    soft_score = (sum(soft_scores) / 5.0) * 30.0
    
    base_score = hard_score + soft_score
    
    # 3. Stuffing Penalty: many skills + shallow descriptions = -30%
    if skills_count > 25 and work_depth < 0.25:
        base_score *= 0.7
        
    return max(0.0, min(100.0, base_score))
