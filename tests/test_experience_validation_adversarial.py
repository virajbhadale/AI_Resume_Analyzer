from src.inference.candidates import CandidateStatus, ExperienceCandidate
from src.inference.validation import ExperienceCandidateValidator


def validate(candidate: ExperienceCandidate):
    return ExperienceCandidateValidator().validate(candidate)


def test_complete_experience_is_validated():
    candidate = ExperienceCandidate(
        candidate_id="c1",
        block_ids=["b1", "b2", "b3"],
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2022",
        end_date="Present",
        is_current=True,
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.VALIDATED
    assert result.is_valid is True
    assert result.confidence >= 0.90


def test_experience_without_company_is_not_rejected():
    candidate = ExperienceCandidate(
        candidate_id="c2",
        block_ids=["b1", "b2"],
        role="Data Scientist",
        start_date="2022",
        end_date="Present",
        is_current=True,
    )

    result = validate(candidate)

    assert result.status != CandidateStatus.REJECTED
    assert result.is_valid is True


def test_experience_without_date_is_not_rejected():
    candidate = ExperienceCandidate(
        candidate_id="c3",
        block_ids=["b1", "b2"],
        role="Data Scientist",
        company="ABC Technologies",
    )

    result = validate(candidate)

    assert result.status != CandidateStatus.REJECTED
    assert result.is_valid is True


def test_role_only_candidate_needs_review():
    candidate = ExperienceCandidate(
        candidate_id="c4",
        block_ids=["b1"],
        role="Research Assistant",
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.NEEDS_REVIEW
    assert result.is_valid is False


def test_empty_candidate_is_rejected():
    candidate = ExperienceCandidate(
        candidate_id="c5",
        block_ids=[],
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.REJECTED
    assert result.is_valid is False


def test_company_only_candidate_is_rejected():
    candidate = ExperienceCandidate(
        candidate_id="c6",
        block_ids=["b1"],
        company="ABC Technologies",
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.REJECTED
    assert result.is_valid is False


def test_date_only_candidate_is_rejected():
    candidate = ExperienceCandidate(
        candidate_id="c7",
        block_ids=["b1"],
        start_date="2022",
        end_date="2024",
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.REJECTED
    assert result.is_valid is False


def test_location_only_candidate_is_rejected():
    candidate = ExperienceCandidate(
        candidate_id="c8",
        block_ids=["b1"],
        location="Pune, India",
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.REJECTED
    assert result.is_valid is False


def test_role_is_more_important_than_metadata_completeness():
    candidate = ExperienceCandidate(
        candidate_id="c9",
        block_ids=["b1"],
        role="Machine Learning Intern",
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.NEEDS_REVIEW
    assert result.is_valid is False
    assert result.confidence > 0.0


def test_validation_does_not_mutate_candidate():
    candidate = ExperienceCandidate(
        candidate_id="c10",
        block_ids=["b1", "b2"],
        role="Data Scientist",
        company="ABC Technologies",
    )

    before = candidate.model_dump()

    validate(candidate)

    after = candidate.model_dump()

    assert after == before