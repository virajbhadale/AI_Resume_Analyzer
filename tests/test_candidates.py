from src.inference.candidates import (
    CandidateStatus,
    ExperienceCandidate,
)


def test_candidate_defaults():

    candidate = ExperienceCandidate(
        candidate_id="candidate-001"
    )

    assert candidate.candidate_id == "candidate-001"
    assert candidate.block_ids == []

    assert candidate.role is None
    assert candidate.company is None
    assert candidate.location is None

    assert candidate.start_date is None
    assert candidate.end_date is None
    assert candidate.is_current is None

    assert candidate.confidence == 0.0
    assert candidate.status == CandidateStatus.CANDIDATE


def test_candidate_with_metadata():

    candidate = ExperienceCandidate(
        candidate_id="candidate-001",
        block_ids=["b1", "b2", "b3"],
        role="AI Engineer",
        company="Marva AI",
        location="Pune, India",
        start_date="2025-01-06",
        end_date="2025-05-17",
        section="experience",
    )

    assert candidate.role == "AI Engineer"
    assert candidate.company == "Marva AI"
    assert candidate.location == "Pune, India"

    assert candidate.start_date == "2025-01-06"
    assert candidate.end_date == "2025-05-17"

    assert candidate.section == "experience"


def test_add_block_prevents_duplicates():

    candidate = ExperienceCandidate(
        candidate_id="candidate-001"
    )

    candidate.add_block("b1")
    candidate.add_block("b2")
    candidate.add_block("b1")

    assert candidate.block_ids == ["b1", "b2"]


def test_candidate_field_helpers():

    candidate = ExperienceCandidate(
        candidate_id="candidate-001",
        role="Data Analyst",
        company="Example Corp",
        location="Mumbai, India",
        start_date="2024-01-01",
        end_date="2024-06-01",
    )

    assert candidate.has_role() is True
    assert candidate.has_company() is True
    assert candidate.has_location() is True
    assert candidate.has_date() is True


def test_candidate_without_fields():

    candidate = ExperienceCandidate(
        candidate_id="candidate-001"
    )

    assert candidate.has_role() is False
    assert candidate.has_company() is False
    assert candidate.has_location() is False
    assert candidate.has_date() is False


def test_candidate_status():

    candidate = ExperienceCandidate(
        candidate_id="candidate-001",
        status=CandidateStatus.NEEDS_REVIEW,
    )

    assert candidate.status == CandidateStatus.NEEDS_REVIEW