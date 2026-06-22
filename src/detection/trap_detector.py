"""
Trap Detector — Identifies honeypot profiles, keyword stuffers, and other adversarial attempts
to game the ranking system, and applies appropriate score penalties.
"""

from typing import Any

def detect_traps(features: dict[str, Any], signal_violations: list[str]) -> tuple[float, list[str]]:
    """
    Analyzes candidate features for traps and adversarial profiles.
    Returns:
        (penalty, reasons)
        where:
        - penalty: float, negative value indicating total score penalty (0 if clean)
        - reasons: list of strings indicating which checks failed
    """
    penalty = 0.0
    reasons = []
    
    # 1. HONEYPOT CHECK 1: Impossible experience timeline
    # Graduation year vs total experience years
    grad_year = features.get("graduation_year")
    total_exp_months = features.get("total_experience_months", 0)
    total_exp_years = total_exp_months / 12.0
    
    if grad_year:
        # Present year of simulation is 2026
        max_possible_years = 2026 - grad_year
        # If total experience exceeds years since graduation + 1.5 years buffer, it's impossible
        if total_exp_years > max_possible_years + 1.5:
            penalty -= 60.0
            reasons.append("impossible_timeline")
            
    # 2. HONEYPOT CHECK 2: Behavioral signal envelope violations
    violation_count = len(signal_violations)
    if violation_count >= 3:
        penalty -= 20.0 * violation_count
        reasons.append(f"signal_violations:{violation_count}")
        
    # 3. KEYWORD STUFFER CHECK
    skills_count = features.get("skills_count", 0)
    work_depth = features.get("work_description_depth", 0.5)
    # More than 25 skills AND shallow work descriptions = likely keyword stuffer
    if skills_count > 25 and work_depth < 0.25:
        penalty -= 30.0
        reasons.append("keyword_stuffer")
        
    # 4. DISQUALIFIER: Only worked at IT consulting / outsourcing firms
    if features.get("only_consulting", False):
        penalty -= 40.0
        reasons.append("consulting_only_career")
        
    # 5. LOGISTICAL GAP: Long notice period (e.g. > 90 days)
    signals = features.get("redrob_signals", {})
    notice_days = signals.get("notice_period_days", 0)
    if notice_days > 90:
        penalty -= 15.0
        reasons.append("excessive_notice_period")
        
    return penalty, reasons
