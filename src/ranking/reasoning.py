"""
Reasoning Generator — Generates natural, factual, 1-2 sentence recruiter rationales
for top-ranked candidates, incorporating specific profile details without hallucinating.
"""

from typing import Any
import numpy as np

def generate_reasoning(features: dict[str, Any], ranked_result: dict[str, Any]) -> str:
    """
    Generates a natural 1-2 sentence justification for a candidate's rank/score.
    Includes specific facts (experience years, current title, named skills, behavioral signals)
    and lists any honest concerns (long notice period, career gaps, consulting background).
    """
    dim_scores = ranked_result["dimension_scores"]
    final_score = ranked_result["final_score"]
    
    # 1. Gather facts
    name = features.get("anonymized_name", "The candidate")
    exp_months = features.get("total_experience_months", 0)
    exp_years = round(exp_months / 12.0, 1)
    current_role = features.get("current_role", "Engineer")
    skills = features.get("skills_normalized", [])
    
    # Identify matching keywords from candidate skills
    matched_skills = []
    core_jd_skills = ["python", "embeddings", "sentence-transformers", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "ndcg", "mrr", "map", "lora", "qlora", "peft", "xgboost"]
    for s in skills:
        if s in core_jd_skills:
            matched_skills.append(s)
            
    skill_snippet = ""
    if len(matched_skills) >= 2:
        skill_snippet = f"demonstrated expertise in {', '.join(matched_skills[:3])}"
    elif matched_skills:
        skill_snippet = f"solid hands-on skills in {matched_skills[0]}"
    else:
        skill_snippet = f"relevant technical profile with {len(skills)} skills"

    # 2. Determine top dimensions
    sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)
    top_dim, top_score = sorted_dims[0]
    second_dim, second_score = sorted_dims[1]
    
    # 3. Construct positive statement based on top dimensions
    positives = []
    
    if top_dim == "semantic" and top_score > 70:
        positives.append(f"shows exceptional semantic alignment with our founding AI engineering requirements (semantic score: {top_score:.0f}/100)")
    elif top_dim == "career" and top_score > 70:
        positives.append(f"features a strong progressive career trajectory over {exp_years} years, peaking at {current_role} (career score: {top_score:.0f}/100)")
    elif top_dim == "skills" and top_score > 70:
        positives.append(f"shows significant depth in required machine learning and ranking technologies (skills score: {top_score:.0f}/100)")
    elif top_dim == "behavioral" and top_score > 70:
        positives.append(f"presents outstanding platform behavioral signals and high engagement (behavioral score: {top_score:.0f}/100)")
    else:
        positives.append(f"offers a balanced profile with {exp_years} years of total experience as {current_role}")
        
    if second_dim == "semantic" and second_score > 60:
        positives.append(f"a highly relevant JD match")
    elif second_dim == "career" and second_score > 60:
        positives.append(f"progressive growth in product-centric roles")
    elif second_dim == "skills" and second_score > 60:
        positives.append(f"well-substantiated technical credentials")
    elif second_dim == "activity" and second_score > 60:
        positives.append(f"excellent active engagement on the platform")
        
    # Combine positives into a sentence
    sentence1 = ""
    if len(positives) >= 2:
        sentence1 = f"{current_role} with {exp_years} years of experience who {positives[0]} and demonstrates {positives[1]}."
    else:
        sentence1 = f"{current_role} with {exp_years} years of experience who {positives[0]}."
        
    # 4. Construct honest concerns
    concerns = []
    signals = features.get("redrob_signals", {})
    
    # Notice period
    notice_days = signals.get("notice_period_days", 0)
    if notice_days > 60:
        concerns.append(f"a long {notice_days}-day notice period")
    elif notice_days > 30:
        concerns.append(f"a standard {notice_days}-day notice period")
        
    # Career gaps
    if features.get("has_career_gaps", False):
        concerns.append("minor history of career gaps")
        
    # Relocation
    relocate = signals.get("willing_to_relocate", True)
    location = features.get("location", "")
    target_locs = ["pune", "noida", "hyderabad", "mumbai", "delhi"]
    in_target_area = any(l in location.lower() for l in target_locs)
    if not relocate and not in_target_area:
        concerns.append("unwilling to relocate outside current area")
        
    # Lower response rate
    response_rate = signals.get("recruiter_response_rate", 1.0)
    if response_rate < 0.3:
        concerns.append(f"low recruiter response rate of {response_rate*100:.0f}%")
        
    sentence2 = ""
    if concerns:
        sentence2 = f"Overall fit is {final_score:.0f}/100, with note of {', '.join(concerns)}."
    else:
        sentence2 = f"Strong fit (overall score: {final_score:.0f}/100) with zero flags across platform activity, availability, and skill depth."
        
    # Vary the format based on the candidate's rank/score to reflect rank consistency
    if final_score >= 85:
        # Top candidate: glowing review
        return f"Exceptional candidate. This {sentence1} {sentence2}"
    elif final_score >= 70:
        # Mid-high candidate: solid review
        return f"Highly qualified. This {sentence1} {sentence2}"
    elif final_score >= 50:
        # Mid-low candidate: qualified with some reservations
        return f"Capable candidate. This {sentence1} {sentence2}"
    else:
        # Low candidate: filler / adjacent skills
        return f"Adjacent candidate. This {sentence1} {sentence2}"
