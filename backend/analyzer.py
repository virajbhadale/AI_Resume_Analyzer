from typing import List, Optional, Dict, Any

def analyze_missing_keywords(
    resume_skills: List[str], 
    target_role: Optional[str],
    roles_config: Dict[str, Any], 
    jd_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares extracted resume skills against target role definitions 
    from configuration to identify critical skill gaps.
    """
    role_data = {}

    if target_role:
        role_data = roles_config.get("roles", {}).get(target_role, {})
    
    resume_lower = {s.lower() for s in resume_skills}
    
    missing_core = [s for s in role_data.get("core_skills", []) if s.lower() not in resume_lower]
    missing_secondary = [s for s in role_data.get("secondary_skills", []) if s.lower() not in resume_lower]
    missing_keywords = [k for k in role_data.get("keywords", []) if k.lower() not in resume_lower]
    
    urgency_score = (len(missing_core) * 10) + (len(missing_secondary) * 5)
    
    return {
        "missing_core": missing_core,
        "missing_secondary": missing_secondary,
        "missing_keywords": missing_keywords,
        "urgency_score": urgency_score
    }