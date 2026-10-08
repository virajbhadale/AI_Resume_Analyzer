"""Evidence Fusion and Verifiable Confidence Scoring."""
from typing import Dict, Any
from src.document_models import ExperienceEpisode, ContactInfo, CanonicalResume

class ConfidenceScorer:
    def __init__(self):
        pass

    def score_experience_episode(self, ep: ExperienceEpisode) -> float:
        """Computes verifiable confidence for an extracted experience episode."""
        score = 0.40  # Base presence score

        # Role clarity (+0.20)
        if ep.role and ep.role != "Role Not Specified":
            score += 0.20

        # Company clarity (+0.15)
        if ep.company and ep.company != "Company Not Specified":
            score += 0.15

        # Dates present (+0.15)
        if ep.start_date:
            score += 0.15

        # Descriptions / achievements present (+0.10)
        if len(ep.bullet_points) >= 2 or len(ep.description) >= 60:
            score += 0.10

        return round(min(0.99, score), 2)

    def score_overall_resume(self, resume: CanonicalResume) -> Dict[str, Any]:
        """Provides section-by-section and overall confidence scores."""
        contact_scores = list(resume.contact.confidence.values())
        contact_conf = round(sum(contact_scores) / len(contact_scores), 2) if contact_scores else 0.50

        exp_scores = [ep.confidence for ep in resume.experience]
        exp_conf = round(sum(exp_scores) / len(exp_scores), 2) if exp_scores else 0.60

        skills_conf = 0.90 if len(resume.skills) >= 5 else 0.70
        edu_conf = 0.90 if resume.education else 0.60

        overall = round((contact_conf * 0.25) + (exp_conf * 0.35) + (skills_conf * 0.20) + (edu_conf * 0.20), 2)

        return {
            "overall_confidence": overall,
            "contact_confidence": contact_conf,
            "experience_confidence": exp_conf,
            "skills_confidence": skills_conf,
            "education_confidence": edu_conf
        }
