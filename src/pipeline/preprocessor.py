"""
Preprocessor — Cleans raw candidate profiles and extracts key features.
"""

from datetime import datetime
from dateutil import parser as date_parser
from typing import Any, Optional

SKILL_ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "llm": "large language model",
    "llms": "large language model",
    "rag": "retrieval-augmented generation",
    "ai": "artificial intelligence",
    "genai": "generative ai",
    "generative ai": "generative ai",
    "tcs": "tata consultancy services",
    "wipro": "wipro",
    "infosys": "infosys",
    "accenture": "accenture",
    "cognizant": "cognizant",
    "capgemini": "capgemini",
}

def normalize_skill_name(name: str) -> str:
    """Normalize a skill name to lowercase, stripped, and map aliases."""
    clean_name = name.lower().strip()
    return SKILL_ALIASES.get(clean_name, clean_name)

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Safely parse a date string into a datetime object."""
    if not date_str:
        return None
    try:
        return date_parser.parse(date_str)
    except Exception:
        return None

def score_work_description_depth(career_history: list[dict]) -> float:
    """
    Estimates how substantive a candidate's work descriptions are.
    Returns 0.0 (shallow/empty) to 1.0 (detailed and specific).
    """
    total_words = 0
    total_roles = 0
    
    for job in career_history:
        description = job.get("description", "")
        if description:
            words = len(description.split())
            total_words += words
        total_roles += 1
    
    if total_roles == 0:
        return 0.0
    
    avg_words_per_role = total_words / total_roles
    # Heuristic: < 20 words per role = shallow, > 100 words = detailed
    depth_score = min(1.0, avg_words_per_role / 100.0)
    return depth_score

def detect_career_gaps(career_history: list[dict]) -> bool:
    """
    Identify unexplained gaps > 6 months (180 days) between roles.
    Assumes career_history is sorted ascending by start_date.
    """
    if len(career_history) <= 1:
        return False
    
    sorted_jobs = sorted(
        [job for job in career_history if job.get("start_date")],
        key=lambda x: parse_date(x["start_date"]) or datetime.min
    )
    
    for i in range(len(sorted_jobs) - 1):
        curr_job = sorted_jobs[i]
        next_job = sorted_jobs[i+1]
        
        curr_end_str = curr_job.get("end_date")
        next_start_str = next_job.get("start_date")
        
        if not curr_end_str or not next_start_str:
            continue
            
        curr_end = parse_date(curr_end_str)
        next_start = parse_date(next_start_str)
        
        if curr_end and next_start:
            gap = (next_start - curr_end).days
            if gap > 180: # > 6 months gap
                return True
                
    return False

def preprocess_candidate(raw_candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Transforms a raw candidate dict into preprocessed features.
    """
    candidate_id = raw_candidate["candidate_id"]
    profile = raw_candidate.get("profile", {})
    career_history = raw_candidate.get("career_history", [])
    education = raw_candidate.get("education", [])
    skills = raw_candidate.get("skills", [])
    signals = raw_candidate.get("redrob_signals", {})
    
    # 1. Total and average experience
    total_exp_months = sum(job.get("duration_months", 0) for job in career_history)
    
    # 2. Career timeline sorted descending (most recent first)
    timeline = sorted(
        career_history,
        key=lambda x: parse_date(x.get("start_date")) or datetime.min,
        reverse=True
    )
    
    # 3. Current title and company
    current_role = profile.get("current_title", "")
    if not current_role and timeline:
        current_role = timeline[0].get("title", "")
        
    # 4. Normalized skills list
    normalized_skills = []
    for skill in skills:
        skill_name = skill.get("name")
        if skill_name:
            normalized_skills.append(normalize_skill_name(skill_name))
    normalized_skills = list(set(normalized_skills)) # deduplicate
    
    # 5. Graduation year (earliest end_year from education)
    grad_years = [edu.get("end_year") for edu in education if edu.get("end_year")]
    earliest_grad_year = min(grad_years) if grad_years else None
    
    # 6. Work description depth
    work_depth = score_work_description_depth(career_history)
    
    # 7. Career gaps detection
    has_gaps = detect_career_gaps(career_history)
    
    # 8. Check if candidate worked ONLY in consulting firms
    # consulting firm names from the list
    consulting_keywords = ["tata consultancy services", "tcs", "wipro", "infosys", "accenture", "cognizant", "capgemini"]
    total_companies = len(career_history)
    consulting_companies_count = 0
    for job in career_history:
        comp_name = job.get("company", "").lower().strip()
        is_consulting = any(k in comp_name for k in consulting_keywords)
        if is_consulting:
            consulting_companies_count += 1
            
    only_consulting = (total_companies > 0 and consulting_companies_count == total_companies)

    return {
        "id": candidate_id,
        "anonymized_name": profile.get("anonymized_name", ""),
        "headline": profile.get("headline", ""),
        "summary": profile.get("summary", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "total_experience_months": total_exp_months,
        "graduation_year": earliest_grad_year,
        "skills_normalized": normalized_skills,
        "skills_count": len(normalized_skills),
        "career_timeline": timeline,
        "current_role": current_role,
        "has_career_gaps": has_gaps,
        "work_description_depth": work_depth,
        "only_consulting": only_consulting,
        "education": education,
        "skills_raw": skills,
        "redrob_signals": signals
    }
