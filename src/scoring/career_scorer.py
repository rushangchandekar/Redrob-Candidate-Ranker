"""
Career Scorer — Evaluates career trajectory, relevance, progression, consistency, and tenure quality.
"""

from typing import Any

# Relevance keywords
ML_AI_KEYWORDS = [
    "ai", "ml", "machine learning", "nlp", "natural language", "deep learning", 
    "computer vision", "embedding", "retrieval", "vector", "search", "ranking", 
    "recommender", "recommendation", "reinforcement", "neural", "transformers",
    "llm", "large language", "generative ai", "genai", "rag"
]

BACKEND_DE_KEYWORDS = [
    "backend", "data engineer", "software engineer", "infrastructure", 
    "python", "sql", "spark", "hadoop", "cloud", "aws", "gcp", "azure", 
    "scala", "java", "analytics engineer", "system architect"
]

def get_title_rank(title: str) -> int:
    """Helper to assign a seniority rank based on job title."""
    title_lower = title.lower()
    if any(kw in title_lower for kw in ["director", "head", "vp", "chief"]):
        return 5
    if any(kw in title_lower for kw in ["lead", "principal", "founding", "architect", "staff"]):
        return 4
    if "senior" in title_lower or "sr" in title_lower:
        return 3
    if any(kw in title_lower for kw in ["junior", "jr", "intern", "associate"]):
        return 1
    # Standard engineer/developer
    return 2

def get_job_relevance_weight(job: dict) -> float:
    """Determines relevance weight of a job role relative to Senior AI Engineer."""
    title = job.get("title", "").lower()
    desc = job.get("description", "").lower()
    
    # 1. Check ML/AI keywords (highest relevance)
    if any(kw in title for kw in ML_AI_KEYWORDS) or any(kw in desc for kw in ML_AI_KEYWORDS):
        # Disqualify CV/Robotics/Speech pure roles if they don't have NLP/IR (from JD: "things we explicitly do not want")
        # However, let's just use 1.0 for general AI/ML relevance
        return 1.0
        
    # 2. Check Backend / Data Engineering keywords (medium relevance)
    if any(kw in title for kw in BACKEND_DE_KEYWORDS) or any(kw in desc for kw in BACKEND_DE_KEYWORDS):
        return 0.5
        
    return 0.1

def calculate_relevant_experience(timeline: list[dict]) -> float:
    """Calculate weighted relevant experience months."""
    relevant_months = 0.0
    for job in timeline:
        duration = job.get("duration_months", 0)
        weight = get_job_relevance_weight(job)
        relevant_months += duration * weight
    return relevant_months

def detect_career_progression(timeline: list[dict]) -> float:
    """
    Evaluates career progression.
    Returns a progression factor between 0.0 and 1.0.
    """
    if not timeline:
        return 0.5
        
    # Chronological order (oldest first)
    chrono_jobs = list(reversed(timeline))
    ranks = [get_title_rank(job.get("title", "")) for job in chrono_jobs]
    
    progression_score = 0.5  # Base score
    
    # 1. Seniority of current/latest role
    latest_rank = ranks[-1] if ranks else 2
    if latest_rank >= 4:
        progression_score += 0.3  # Lead/Principal/Founding
    elif latest_rank == 3:
        progression_score += 0.2  # Senior
        
    # 2. Check for rank increase over time
    if len(ranks) >= 2:
        increases = 0
        for i in range(len(ranks) - 1):
            if ranks[i+1] > ranks[i]:
                increases += 1
        if increases > 0:
            progression_score += min(0.2, increases * 0.1)
            
    return min(1.0, progression_score)

def score_career_trajectory(features: dict[str, Any], jd_profile: dict[str, Any]) -> float:
    """
    Scores career trajectory based on:
    1. Relevant experience (0-40 points)
    2. Career progression (0-30 points)
    3. Domain consistency (0-20 points)
    4. Tenure quality & hopping penalties (0 to -10 points)
    """
    timeline = features.get("career_timeline", [])
    if type(timeline).__name__ == 'ndarray':
        timeline = list(timeline)
    if timeline is None or len(timeline) == 0:
        return 0.0
        
    score = 0.0
    
    # 1. Relevant experience weight (0-40 points)
    relevant_months = calculate_relevant_experience(timeline)
    # 4 years of weighted relevant experience = full 40 points
    score += min(40.0, (relevant_months / 48.0) * 40.0)
    
    # 2. Career progression (0-30 points)
    progression = detect_career_progression(timeline)
    score += progression * 30.0
    
    # 3. Domain consistency (0-20 points)
    total_months = sum(job.get("duration_months", 0) for job in timeline)
    if total_months > 0:
        domain_ratio = relevant_months / total_months
    else:
        domain_ratio = 0.0
    score += domain_ratio * 20.0
    
    # 4. Tenure penalty (short stints and hopping)
    short_stint_count = sum(1 for job in timeline if job.get("duration_months", 12) < 6)
    stint_penalty = min(10.0, short_stint_count * 3.0)
    score -= stint_penalty
    
    # Job hopping / title-chasing penalty (average tenure < 18 months)
    total_roles = len(timeline)
    if total_roles > 0:
        avg_tenure = total_months / total_roles
        if avg_tenure < 18.0:
            score -= 10.0 # penalty for frequent switching
            
    return max(0.0, min(100.0, score))
