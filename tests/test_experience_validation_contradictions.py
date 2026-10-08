from src.inference.candidates import CandidateStatus, ExperienceCandidate
from src.inference.evidence_models import (
    EvidencePolarity,
    EvidenceSignal,
    EvidenceType,
)
from src.inference.validation import ExperienceCandidateValidator


def validate(candidate: ExperienceCandidate):
    return ExperienceCandidateValidator().validate(candidate)


def add_signal(
    candidate: ExperienceCandidate,
    *,
    signal_type: EvidenceType,
    value,
    confidence: float,
    polarity: EvidencePolarity,
    source_block_id: str,
    source_field: str | None = None,
    explanation: str | None = None,
):
    candidate.evidence.add_signal(
        EvidenceSignal(
            signal_type=signal_type,
            value=value,
            confidence=confidence,
            polarity=polarity,
            source_block_id=source_block_id,
            source_field=source_field,
            explanation=explanation,
        )
    )


def test_strong_positive_evidence_is_validated():
    candidate = ExperienceCandidate(
        candidate_id="c1",
        block_ids=["b1", "b2", "b3"],
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2022",
        end_date="Present",
        is_current=True,
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.ROLE,
        value="Data Scientist",
        confidence=0.95,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b1",
        source_field="role",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.COMPANY,
        value="ABC Technologies",
        confidence=0.90,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b2",
        source_field="company",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.DATE,
        value={
            "start_date": "2022",
            "end_date": "Present",
            "is_current": True,
        },
        confidence=0.95,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b3",
        source_field="date",
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.VALIDATED
    assert result.is_valid is True
    assert len(result.positive_evidence) == 3
    assert len(result.negative_evidence) == 0


def test_contradictory_evidence_is_preserved():
    candidate = ExperienceCandidate(
        candidate_id="c2",
        block_ids=["b1", "b2"],
        role="Data Scientist",
        company="ABC Technologies",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.ROLE,
        value="Data Scientist",
        confidence=0.95,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b1",
        source_field="role",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.SEMANTIC,
        value="description_like_text",
        confidence=0.70,
        polarity=EvidencePolarity.CONTRADICT,
        source_block_id="b2",
        explanation="Block resembles descriptive content rather than an experience header.",
    )

    result = validate(candidate)

    assert len(result.positive_evidence) == 1
    assert len(result.negative_evidence) == 1

    assert result.positive_evidence[0].polarity == EvidencePolarity.SUPPORT
    assert result.negative_evidence[0].polarity == EvidencePolarity.CONTRADICT


def test_neutral_evidence_does_not_become_negative_evidence():
    candidate = ExperienceCandidate(
        candidate_id="c3",
        block_ids=["b1"],
        role="Data Scientist",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.LAYOUT,
        value="normal_layout",
        confidence=0.80,
        polarity=EvidencePolarity.NEUTRAL,
        source_block_id="b1",
    )

    result = validate(candidate)

    assert len(result.positive_evidence) == 0
    assert len(result.negative_evidence) == 0


def test_strong_valid_experience_survives_irrelevant_negative_evidence():
    candidate = ExperienceCandidate(
        candidate_id="c4",
        block_ids=["b1", "b2", "b3", "b4"],
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2022",
        end_date="Present",
        is_current=True,
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.ROLE,
        value="Data Scientist",
        confidence=0.95,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b1",
        source_field="role",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.COMPANY,
        value="ABC Technologies",
        confidence=0.90,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b2",
        source_field="company",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.DATE,
        value={
            "start_date": "2022",
            "end_date": "Present",
            "is_current": True,
        },
        confidence=0.95,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b3",
        source_field="date",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.SEMANTIC,
        value="sentence_like_description",
        confidence=0.60,
        polarity=EvidencePolarity.CONTRADICT,
        source_block_id="b4",
        explanation="Description block is sentence-like.",
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.VALIDATED
    assert result.is_valid is True
    assert len(result.positive_evidence) == 3
    assert len(result.negative_evidence) == 1


def test_role_only_candidate_with_contradiction_needs_review():
    candidate = ExperienceCandidate(
        candidate_id="c5",
        block_ids=["b1"],
        role="Data Scientist",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.ROLE,
        value="Data Scientist",
        confidence=0.70,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b1",
        source_field="role",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.SEMANTIC,
        value="prose_like",
        confidence=0.80,
        polarity=EvidencePolarity.CONTRADICT,
        source_block_id="b1",
        explanation="The block also resembles prose.",
    )

    result = validate(candidate)

    assert result.status == CandidateStatus.NEEDS_REVIEW
    assert result.is_valid is False
    assert len(result.positive_evidence) == 1
    assert len(result.negative_evidence) == 1


def test_contradictory_evidence_does_not_mutate_candidate():
    candidate = ExperienceCandidate(
        candidate_id="c6",
        block_ids=["b1", "b2"],
        role="Data Scientist",
        company="ABC Technologies",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.ROLE,
        value="Data Scientist",
        confidence=0.90,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b1",
        source_field="role",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.SEMANTIC,
        value="prose",
        confidence=0.80,
        polarity=EvidencePolarity.CONTRADICT,
        source_block_id="b2",
    )

    before = candidate.model_dump()

    validate(candidate)

    after = candidate.model_dump()

    assert after == before


def test_multiple_negative_signals_are_all_preserved():
    candidate = ExperienceCandidate(
        candidate_id="c7",
        block_ids=["b1", "b2", "b3"],
        role="Data Scientist",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.SEMANTIC,
        value="prose",
        confidence=0.70,
        polarity=EvidencePolarity.CONTRADICT,
        source_block_id="b1",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.STRUCTURAL,
        value="missing_experience_context",
        confidence=0.60,
        polarity=EvidencePolarity.CONTRADICT,
        source_block_id="b2",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.SECTION,
        value="projects",
        confidence=0.90,
        polarity=EvidencePolarity.CONTRADICT,
        source_block_id="b3",
    )

    result = validate(candidate)

    assert len(result.negative_evidence) == 3


def test_support_and_neutral_evidence_do_not_create_false_negative():
    candidate = ExperienceCandidate(
        candidate_id="c8",
        block_ids=["b1"],
        role="Research Assistant",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.ROLE,
        value="Research Assistant",
        confidence=0.85,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b1",
        source_field="role",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.LAYOUT,
        value="sidebar",
        confidence=0.70,
        polarity=EvidencePolarity.NEUTRAL,
        source_block_id="b1",
    )

    result = validate(candidate)

    assert len(result.positive_evidence) == 1
    assert len(result.negative_evidence) == 0
    assert result.status == CandidateStatus.NEEDS_REVIEW


def test_validation_reasons_are_present_for_contradictory_candidate():
    candidate = ExperienceCandidate(
        candidate_id="c9",
        block_ids=["b1"],
        role="Data Scientist",
    )

    add_signal(
        candidate,
        signal_type=EvidenceType.SEMANTIC,
        value="prose",
        confidence=0.80,
        polarity=EvidencePolarity.CONTRADICT,
        source_block_id="b1",
    )

    result = validate(candidate)

    assert result.status in {
        CandidateStatus.NEEDS_REVIEW,
        CandidateStatus.REJECTED,
    }

    assert result.reasons