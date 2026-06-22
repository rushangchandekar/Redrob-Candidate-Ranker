"""
Behavioral Scorer — Scores the candidate based on 23 platform activity and engagement signals,
and detects envelope violations that indicate synthetic/honeypot trap profiles.
"""

from typing import Any
import datetime
from dateutil import parser as date_parser

def score_behavioral_signals(features: dict[str, Any], jd_profile: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Evaluates all 23 behavioral signals.
    Returns:
        (score, violations_list)
        where:
        - score: 0 to 100 based on platform activity
        - violations_list: list of strings indicating any signal envelope violations
    """
    signals = features.get("redrob_signals", {})
    violations = []
    
    if not signals:
        return 50.0, ["missing_signals"]
        
    total_weight = 0.0
    weighted_score = 0.0
    
    # Define signal check rules: (min_val, max_val, weight, is_higher_better)
    # None indicates no limit
    continuous_rules = {
        "profile_completeness_score": (0.0, 100.0, 0.8, True),
        "recruiter_response_rate": (0.0, 1.0, 1.0, True),
        "avg_response_time_hours": (0.0, 720.0, 0.8, False), # 30 days max
        "connection_count": (0.0, None, 0.4, True),
        "endorsements_received": (0.0, None, 0.5, True),
        "notice_period_days": (0.0, 180.0, 0.7, False), # Shorter notice is better
        "github_activity_score": (-1.0, 100.0, 0.9, True),
        "search_appearance_30d": (0.0, None, 0.4, True),
        "saved_by_recruiters_30d": (0.0, None, 0.6, True),
        "interview_completion_rate": (0.0, 1.0, 0.9, True),
        "offer_acceptance_rate": (-1.0, 1.0, 0.7, True),
        "profile_views_received_30d": (0.0, None, 0.4, True),
        "applications_submitted_30d": (0.0, None, 0.3, True)
    }
    
    for signal_name, rule in continuous_rules.items():
        if signal_name not in signals:
            continue
            
        value = signals[signal_name]
        min_limit, max_limit, weight, higher_better = rule
        
        # 1. Envelope checking
        if min_limit is not None and value < min_limit:
            violations.append(f"{signal_name}_underflow")
        if max_limit is not None and value > max_limit:
            violations.append(f"{signal_name}_overflow")
            
        # 2. Score calculation
        # Normalize to 0-1
        if signal_name == "github_activity_score":
            if value == -1:
                norm_val = 0.1 # default neutral-low if no github linked
            else:
                norm_val = 0.2 + 0.8 * (max(0.0, min(100.0, value)) / 100.0)
        elif signal_name == "offer_acceptance_rate":
            if value == -1:
                norm_val = 0.4 # neutral if no history
            else:
                norm_val = max(0.0, min(1.0, value))
        else:
            # Standard normalization
            low_val = min_limit if min_limit is not None else 0.0
            high_val = max_limit if max_limit is not None else 100.0
            if high_val == low_val:
                norm_val = 1.0
            else:
                norm_val = (value - low_val) / (high_val - low_val)
                norm_val = max(0.0, min(1.0, norm_val))
                if not higher_better:
                    norm_val = 1.0 - norm_val
                    
        weighted_score += norm_val * weight
        total_weight += weight
        
    # Check categorical/boolean signals
    boolean_rules = {
        "open_to_work_flag": (0.8, True),
        "willing_to_relocate": (0.5, True),
        "verified_email": (0.3, True),
        "verified_phone": (0.3, True),
        "linkedin_connected": (0.4, True)
    }
    
    for signal_name, (weight, preferred_val) in boolean_rules.items():
        if signal_name not in signals:
            continue
        value = signals[signal_name]
        
        # Envelope check (must be boolean)
        if not isinstance(value, bool):
            violations.append(f"{signal_name}_not_boolean")
            continue
            
        norm_val = 1.0 if value == preferred_val else 0.2
        
        # Specific relocation check: if location is in target locations, willingness doesn't matter as much.
        # But if candidate is outside, willingness to relocate is highly valued.
        if signal_name == "willing_to_relocate" and not value:
            # Check if location is already in India (or Noida/Pune)
            cand_loc = features.get("location", "").lower()
            target_locs = jd_profile.get("role_signals", {}).get("locations", [])
            in_target_location = any(loc in cand_loc for loc in target_locs)
            if in_target_location:
                norm_val = 0.8 # Less penalty if already in Pune/Noida/Hyderabad
                
        weighted_score += norm_val * weight
        total_weight += weight

    # Work mode preference (categorical)
    pref_work_mode = signals.get("preferred_work_mode")
    if pref_work_mode:
        if pref_work_mode not in ["remote", "hybrid", "onsite", "flexible"]:
            violations.append("preferred_work_mode_invalid_enum")
        else:
            # Role is hybrid, so hybrid/flexible are best; remote/onsite are fine but hybrid is preferred
            weight = 0.5
            if pref_work_mode in ["hybrid", "flexible"]:
                norm_val = 1.0
            else:
                norm_val = 0.7 # small penalty for remote/onsite pure preference
            weighted_score += norm_val * weight
            total_weight += weight
            
    # Skill assessment scores vs stated proficiency cross-check
    # If candidate states "expert" but score is extremely low (e.g. < 20), it's a violation/honeypot!
    skills = features.get("skills_raw", [])
    assessment_scores = signals.get("skill_assessment_scores", {})
    if isinstance(assessment_scores, dict):
        for skill_name, score in assessment_scores.items():
            if score is None:
                continue
            if score < 0 or score > 100:
                violations.append("skill_assessment_score_out_of_bounds")
            
            # Find stated proficiency
            for s in skills:
                if s.get("name", "").lower().strip() == skill_name.lower().strip():
                    prof = s.get("proficiency", "beginner").lower()
                    if prof == "expert" and score < 30:
                        violations.append(f"skill_assessment_mismatch_{skill_name}")
                    if prof == "advanced" and score < 20:
                        violations.append(f"skill_assessment_mismatch_{skill_name}")
                        
    # Check signup_date and last_active_date formats and chronology
    signup_str = signals.get("signup_date")
    active_str = signals.get("last_active_date")
    signup_date = date_parser.parse(signup_str) if signup_str else None
    active_date = date_parser.parse(active_str) if active_str else None
    
    if signup_str and not signup_date:
        violations.append("signup_date_invalid")
    if active_str and not active_date:
        violations.append("last_active_date_invalid")
    if signup_date and active_date and active_date < signup_date:
        violations.append("last_active_before_signup")
        
    # Recency check (availability / activity): if inactive for > 6 months (180 days), down-weight heavily
    # Present date in dataset context is roughly mid 2026 (local time is June 2026)
    if active_date:
        ref_date = datetime.datetime(2026, 6, 22)
        days_inactive = (ref_date - active_date).days
        if days_inactive > 180:
            # inactive for more than 6 months: apply activity down-weighting in the scorer
            # let's add a penalty or adjust the score
            pass
            
    if total_weight == 0:
        return 50.0, violations
        
    final_score = (weighted_score / total_weight) * 100.0
    
    # If inactive for more than 6 months, scale down the score by 30%
    if active_date:
        ref_date = datetime.datetime(2026, 6, 22)
        days_inactive = (ref_date - active_date).days
        if days_inactive > 180:
            final_score *= 0.7
            
    return final_score, violations
