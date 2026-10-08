from src.document_models import (
    ExperienceEpisode,
    CanonicalResume,
    ContactInfo,
)
from src.inference.confidence import ConfidenceScorer


def test_experience_full_confidence():
    scorer = ConfidenceScorer()

    episode = ExperienceEpisode(
        id="exp_001",
        role="AI Engineer Intern",
        company="Marva AI",
        location="Pune, Maharashtra",
        start_date="06 Jan 2025",
        end_date="17 May 2025",
        bullet_points=[
            "Built machine learning pipelines.",
            "Developed RAG-based applications.",
        ],
        description=(
            "Built and deployed AI applications using "
            "Python, LangChain and machine learning."
        ),
    )

    score = scorer.score_experience_episode(episode)

    assert score == 1.0 or score == 0.99


def test_experience_role_company_dates():
    scorer = ConfidenceScorer()

    episode = ExperienceEpisode(
        id="exp_002",
        role="Data Analyst",
        company="ComplyHappy",
        start_date="Jun 2023",
    )

    score = scorer.score_experience_episode(episode)

    assert score == 0.90


def test_experience_missing_role_and_company():
    scorer = ConfidenceScorer()

    episode = ExperienceEpisode(
        id="exp_003",
        role="Role Not Specified",
        company="Company Not Specified",
    )

    score = scorer.score_experience_episode(episode)

    assert score == 0.40


def test_experience_with_description():
    scorer = ConfidenceScorer()

    episode = ExperienceEpisode(
        id="exp_004",
        role="ML Engineer",
        company="Apex AI",
        description="A" * 60,
    )

    score = scorer.score_experience_episode(episode)

    assert score == 0.85