from src.document_models import ExperienceEpisode
from src.inference.confidence import ConfidenceScorer


def make_episode(
    role="Role Not Specified",
    company="Company Not Specified",
    start_date=None,
    end_date=None,
    bullet_points=None,
    description="",
    episode_id="test-episode",
):
    return ExperienceEpisode(
        id=episode_id,
        role=role,
        company=company,
        start_date=start_date,
        end_date=end_date,
        bullet_points=bullet_points or [],
        description=description,
    )


def test_complete_experience_has_strong_final_confidence():
    scorer = ConfidenceScorer()

    episode = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    confidence = scorer.score_experience_episode(episode)

    assert confidence >= 0.90
    assert confidence <= 0.99


def test_missing_company_lowers_final_confidence():
    scorer = ConfidenceScorer()

    complete = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    missing_company = make_episode(
        role="Data Scientist",
        company="Company Not Specified",
        start_date="2023",
        end_date="2024",
    )

    complete_confidence = scorer.score_experience_episode(complete)
    missing_company_confidence = scorer.score_experience_episode(missing_company)

    assert complete_confidence > missing_company_confidence


def test_missing_date_lowers_final_confidence():
    scorer = ConfidenceScorer()

    complete = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    missing_date = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
    )

    complete_confidence = scorer.score_experience_episode(complete)
    missing_date_confidence = scorer.score_experience_episode(missing_date)

    assert complete_confidence > missing_date_confidence


def test_role_only_experience_has_lower_confidence():
    scorer = ConfidenceScorer()

    role_only = make_episode(
        role="Data Scientist",
    )

    complete = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    assert (
        scorer.score_experience_episode(role_only)
        < scorer.score_experience_episode(complete)
    )


def test_description_evidence_contributes_confidence():
    scorer = ConfidenceScorer()

    without_description = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    with_description = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
        description=(
            "Developed machine learning models and automated analytical "
            "pipelines for production data processing and reporting."
        ),
    )

    assert (
        scorer.score_experience_episode(with_description)
        > scorer.score_experience_episode(without_description)
    )


def test_confidence_is_capped_at_099():
    scorer = ConfidenceScorer()

    episode = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
        bullet_points=[
            "Built machine learning models.",
            "Developed automated pipelines.",
        ],
        description=(
            "Developed machine learning models and automated analytical "
            "pipelines for production data processing and reporting."
        ),
    )

    confidence = scorer.score_experience_episode(episode)

    assert confidence <= 0.99


def test_confidence_is_never_below_zero():
    scorer = ConfidenceScorer()

    episode = make_episode()

    confidence = scorer.score_experience_episode(episode)

    assert confidence >= 0.0


def test_company_not_specified_does_not_receive_company_confidence():
    scorer = ConfidenceScorer()

    missing_company = make_episode(
        role="Data Scientist",
        company="Company Not Specified",
        start_date="2023",
        end_date="2024",
    )

    complete_company = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    missing_confidence = scorer.score_experience_episode(missing_company)
    complete_confidence = scorer.score_experience_episode(complete_company)

    assert complete_confidence > missing_confidence


def test_role_not_specified_does_not_receive_role_confidence():
    scorer = ConfidenceScorer()

    missing_role = make_episode(
        role="Role Not Specified",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    no_role = make_episode(
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    assert (
        scorer.score_experience_episode(missing_role)
        == scorer.score_experience_episode(no_role)
    )


def test_final_confidence_is_deterministic():
    scorer = ConfidenceScorer()

    episode = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
        bullet_points=[
            "Built machine learning models.",
            "Developed automated pipelines.",
        ],
    )

    first = scorer.score_experience_episode(episode)
    second = scorer.score_experience_episode(episode)

    assert first == second