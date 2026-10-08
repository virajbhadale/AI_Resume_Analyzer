from src.inference.candidates import (
    CandidateStatus,
    ExperienceCandidate,
)
from src.inference.validation import (
    ExperienceCandidateValidator,
)

from src.inference.evidence_models import (
    EvidencePolarity,
    EvidenceSignal,
    EvidenceType,
)


def make_candidate(
    *,
    role=None,
    company=None,
    location=None,
    start_date=None,
    end_date=None,
    is_current=None,
    block_ids=None,
):
    return ExperienceCandidate(
        candidate_id="candidate_001",
        block_ids=block_ids or ["b1"],
        role=role,
        company=company,
        location=location,
        start_date=start_date,
        end_date=end_date,
        is_current=is_current,
    )


def test_role_company_date_is_validated():
    candidate = make_candidate(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2022",
        end_date="2024",
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.VALIDATED
    assert result.is_valid is True
    assert "role" in result.positive_evidence
    assert "company" in result.positive_evidence
    assert "date" in result.positive_evidence


def test_role_and_date_is_validated_without_company():
    candidate = make_candidate(
        role="Data Analyst",
        start_date="2023",
        end_date="2024",
        location="Pune, India",
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.VALIDATED
    assert result.is_valid is True
    assert "missing_company" in result.negative_evidence


def test_role_and_company_is_validated_without_date():
    candidate = make_candidate(
        role="Machine Learning Engineer",
        company="ABC Technologies",
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.VALIDATED
    assert result.is_valid is True
    assert "missing_date" in result.negative_evidence


def test_role_only_needs_review():
    candidate = make_candidate(
        role="Data Analyst",
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.NEEDS_REVIEW
    assert result.is_valid is False


def test_missing_role_is_rejected():
    candidate = make_candidate(
        company="ABC Technologies",
        start_date="2023",
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.REJECTED
    assert result.is_valid is False


def test_empty_candidate_is_rejected():
    candidate = make_candidate(
        block_ids=[],
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.REJECTED
    assert result.is_valid is False


def test_validation_does_not_modify_candidate():
    candidate = make_candidate(
        role="Data Analyst",
        company="ABC Technologies",
        start_date="2023",
        end_date="2024",
    )

    original = candidate.model_dump()

    ExperienceCandidateValidator().validate(
        candidate
    )

    assert candidate.model_dump() == original

def test_validation_preserves_supporting_evidence():
    candidate = make_candidate(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
    )

    candidate.evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.ROLE,
            value="Data Scientist",
            confidence=0.95,
            polarity=EvidencePolarity.SUPPORT,
            source_block_id="b1",
            source_field="role",
            explanation="Strong role evidence.",
        )
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.VALIDATED
    assert "role" in result.positive_evidence


def test_validation_preserves_contradictory_evidence():
    candidate = make_candidate(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
    )

    candidate.evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.STRUCTURAL,
            value="bullet",
            confidence=0.80,
            polarity=EvidencePolarity.CONTRADICT,
            source_block_id="b1",
            explanation="Block appears to be list content.",
        )
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.VALIDATED
    assert "structural" in result.negative_evidence


def test_neutral_evidence_is_not_positive_or_negative():
    candidate = make_candidate(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
    )

    candidate.evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.LAYOUT,
            value="sidebar",
            confidence=0.50,
            polarity=EvidencePolarity.NEUTRAL,
            source_block_id="b1",
        )
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert "layout" not in result.positive_evidence
    assert "layout" not in result.negative_evidence


def test_evidence_provenance_is_preserved():
    candidate = make_candidate(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
    )

    signal = EvidenceSignal(
        signal_type=EvidenceType.ROLE,
        value="Data Scientist",
        confidence=0.95,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b17",
        source_field="role",
        explanation="Role detected from heading.",
    )

    candidate.evidence.add_signal(signal)

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert candidate.evidence.signals[0].source_block_id == "b17"
    assert candidate.evidence.signals[0].source_field == "role"
    assert (
        candidate.evidence.signals[0].explanation
        == "Role detected from heading."
    )

    assert result.status == CandidateStatus.VALIDATED


def test_mixed_support_and_contradiction_is_reported():
    candidate = make_candidate(
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2023",
    )

    candidate.evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.ROLE,
            value="Data Scientist",
            confidence=0.95,
            polarity=EvidencePolarity.SUPPORT,
            source_block_id="b1",
        )
    )

    candidate.evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.STRUCTURAL,
            value="bullet",
            confidence=0.70,
            polarity=EvidencePolarity.CONTRADICT,
            source_block_id="b2",
        )
    )

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert "role" in result.positive_evidence
    assert "structural" in result.negative_evidence
    assert result.status == CandidateStatus.VALIDATED    