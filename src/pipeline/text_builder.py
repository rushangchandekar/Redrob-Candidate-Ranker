"""
Text Corpus Builder — Converts preprocessed candidate features into a single rich text representation for semantic embeddings.
"""

from typing import Any

def build_candidate_text(candidate: dict[str, Any]) -> str:
    """
    Creates a detailed text representation of a candidate for embedding.
    Ensures that semantic search captures:
    - Recent roles, companies, and details
    - Normalized and original skills
    - Degrees, fields of study, institutions
    - Professional summary/headline
    """
    parts = []
    
    # 1. Headline & Summary
    headline = candidate.get("headline", "")
    if headline:
        parts.append(f"Professional Headline: {headline}")
        
    summary = candidate.get("summary", "")
    if summary:
        parts.append(f"Summary: {summary}")
        
    # 2. Career History (first 3 roles, which are the most recent as timeline is sorted desc)
    timeline = candidate.get("career_timeline", [])
    for i, job in enumerate(timeline[:3]):
        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")
        industry = job.get("industry", "")
        duration = job.get("duration_months", 0)
        
        role_desc = f"Role {i+1}: {title} at {company} ({duration} months, Industry: {industry})."
        if desc:
            role_desc += f" Responsibilities and achievements: {desc}"
        parts.append(role_desc)
        
    # 3. Skills
    skills = candidate.get("skills_normalized", [])
    if skills:
        parts.append("Technical Skills and Competencies: " + ", ".join(skills))
        
    # 4. Education
    education = candidate.get("education", [])
    for edu in education:
        inst = edu.get("institution", "")
        deg = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        tier = edu.get("tier", "")
        edu_desc = f"Education: {deg} in {field} from {inst}"
        if tier and tier != "unknown":
            edu_desc += f" (Institution Rank: {tier})"
        parts.append(edu_desc)
        
    return " | ".join(filter(None, parts))
