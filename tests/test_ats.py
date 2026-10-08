"""Tests for ATS Technical Analysis and Scoring."""
import pytest
from src.ats.analyzer import ATSAnalyzer
from src.document_models import CanonicalResume, ContactInfo, ExperienceEpisode

def test_ats_analysis_scoring():
    analyzer = ATSAnalyzer()
    resume = CanonicalResume(
        contact=ContactInfo(email="john.doe@email.com", phone="+1 555 123 4567"),
        skills=["Python", "SQL", "Pandas", "NumPy", "Scikit-Learn", "Machine Learning", "Statistics"],
        experience=[
            ExperienceEpisode(
                id="exp1",
                role="Data Scientist",
                company="Alpha Corp",
                start_date="2022",
                end_date="Present",
                description="Built predictive churn model in Python and Scikit-Learn achieving 22% improvement.",
                bullet_points=["Built predictive churn model achieving 22% improvement."]
            )
        ]
    )

    result = analyzer.analyze(resume, target_role_id="data_scientist")
    assert result.target_role == "Data Scientist"
    assert result.overall_score > 60.0
    assert "Python" in result.detected_core_skills
    assert "SQL" in result.detected_core_skills
    assert len(result.actionable_recommendations) >= 1
