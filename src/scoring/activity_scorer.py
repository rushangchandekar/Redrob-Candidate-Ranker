"""
Activity Scorer — Evaluates platform activity, profile completeness, search visibility, 
and external activity (GitHub) to score active engagement.
"""

from typing import Any
import datetime
from dateutil import parser as date_parser

def score_platform_activity(features: dict[str, Any]) -> float:
    """
    Scores candidate platform activity from 0 to 100.
    Based on:
    - Profile completeness (30%)
    - Login recency (20%)
    - Recruiter interest (saves, views, appearances) (30%)
    - External activity (GitHub activity) (20%)
    """
    signals = features.get("redrob_signals", {})
    if not signals:
        return 0.0
        
    score = 0.0
    
    # 1. Profile completeness (30%)
    completeness = signals.get("profile_completeness_score", 0.0)
    score += min(100.0, max(0.0, completeness)) * 0.3
    
    # 2. Login recency (20%)
    active_str = signals.get("last_active_date")
    recency_score = 0.0
    if active_str:
        try:
            active_date = date_parser.parse(active_str)
            ref_date = datetime.datetime(2026, 6, 22) # current simulation date
            days_inactive = (ref_date - active_date).days
            
            if days_inactive <= 7:
                recency_score = 100.0
            elif days_inactive <= 30:
                recency_score = 90.0
            elif days_inactive <= 90:
                recency_score = 70.0
            elif days_inactive <= 180:
                recency_score = 40.0
            else:
                recency_score = 10.0
        except Exception:
            recency_score = 50.0
    else:
        recency_score = 30.0
        
    score += recency_score * 0.2
    
    # 3. Recruiter interest (30%)
    # - saved_by_recruiters_30d: 15% (e.g. 5+ saves = full credit)
    # - search_appearance_30d: 10% (e.g. 200+ appearances = full credit)
    # - profile_views_received_30d: 5% (e.g. 50+ views = full credit)
    saves = signals.get("saved_by_recruiters_30d", 0)
    saves_score = min(100.0, (saves / 5.0) * 100.0)
    
    appearances = signals.get("search_appearance_30d", 0)
    apps_score = min(100.0, (appearances / 200.0) * 100.0)
    
    views = signals.get("profile_views_received_30d", 0)
    views_score = min(100.0, (views / 50.0) * 100.0)
    
    interest_score = (saves_score * 0.5) + (apps_score * 0.33) + (views_score * 0.17)
    score += interest_score * 0.3
    
    # 4. External activity (GitHub) (20%)
    github = signals.get("github_activity_score", -1)
    if github == -1:
        # No GitHub linked: give a neutral-low score of 30
        github_score = 30.0
    else:
        github_score = min(100.0, max(0.0, github))
        
    score += github_score * 0.2
    
    return max(0.0, min(100.0, score))
