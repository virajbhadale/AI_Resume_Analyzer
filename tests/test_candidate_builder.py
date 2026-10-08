from src.inference.anchors import AnchorCandidate
from src.inference.candidate_builder import (
    CandidateBuilder,
)
from src.inference.candidates import (
    CandidateStatus,
)
from src.inference.evidence_models import (
    AnchorStrength,
    AnchorType,
    EvidenceType,
)


def test_build_role_company_date_candidate():

    anchor = AnchorCandidate(
        block_id="block_001",
        text="AI Engineer - Marva AI",
        role="AI Engineer",
        company="Marva AI",
        start_date="2025-01-06",
        end_date="2025-05-17",
        is_current=False,
        location="Pune, India",
        role_confidence=0.95,
        company_confidence=0.92,
        date_confidence=0.96,
        is_anchor_head=True,
    )

    builder = CandidateBuilder()

    candidate = builder.build_candidate(anchor)

    assert candidate.role == "AI Engineer"
    assert candidate.company == "Marva AI"
    assert candidate.location == "Pune, India"

    assert candidate.start_date == "2025-01-06"
    assert candidate.end_date == "2025-05-17"

    assert candidate.status == (
        CandidateStatus.CANDIDATE
    )

    assert "block_001" in candidate.block_ids


def test_anchor_evidence_type():

    anchor = AnchorCandidate(
        block_id="block_001",
        text="Data Analyst",
        role="Data Analyst",
        start_date="2024-01-01",
        end_date="2024-06-01",
        role_confidence=0.90,
        date_confidence=0.95,
    )

    builder = CandidateBuilder()

    evidence = builder.build_anchor_evidence(
        anchor
    )

    assert evidence.anchor_type == (
        AnchorType.ROLE_DATE
    )

    assert evidence.strength == (
        AnchorStrength.MEDIUM
    )


def test_candidate_contains_role_evidence():

    anchor = AnchorCandidate(
        block_id="block_001",
        text="Machine Learning Engineer",
        role="Machine Learning Engineer",
        role_confidence=0.94,
    )

    builder = CandidateBuilder()

    candidate = builder.build_candidate(anchor)

    role_signals = (
        candidate.evidence.signals_for(
            EvidenceType.ROLE
        )
    )

    assert len(role_signals) == 1
    assert role_signals[0].value == (
        "Machine Learning Engineer"
    )


def test_candidate_contains_company_evidence():

    anchor = AnchorCandidate(
        block_id="block_001",
        text="Google",
        company="Google",
        company_confidence=0.93,
    )

    builder = CandidateBuilder()

    candidate = builder.build_candidate(anchor)

    company_signals = (
        candidate.evidence.signals_for(
            EvidenceType.COMPANY
        )
    )

    assert len(company_signals) == 1
    assert company_signals[0].value == "Google"


def test_candidate_contains_date_evidence():

    anchor = AnchorCandidate(
        block_id="block_001",
        text="2024 - Present",
        start_date="2024-01-01",
        end_date=None,
        is_current=True,
        date_confidence=0.96,
    )

    builder = CandidateBuilder()

    candidate = builder.build_candidate(anchor)

    date_signals = (
        candidate.evidence.signals_for(
            EvidenceType.DATE
        )
    )

    assert len(date_signals) == 1

    assert date_signals[0].value[
        "start_date"
    ] == "2024-01-01"

    assert date_signals[0].value[
        "is_current"
    ] is True


def test_candidate_ids_are_unique():

    anchor = AnchorCandidate(
        block_id="block_001",
        text="Data Engineer",
        role="Data Engineer",
        role_confidence=0.90,
    )

    builder = CandidateBuilder()

    first = builder.build_candidate(anchor)
    second = builder.build_candidate(anchor)

    assert first.candidate_id != second.candidate_id


def test_build_multiple_candidates():

    anchors = [
        AnchorCandidate(
            block_id="block_001",
            text="Data Analyst",
            role="Data Analyst",
            company="ComplyHappy",
            role_confidence=0.90,
            company_confidence=0.90,
        ),
        AnchorCandidate(
            block_id="block_002",
            text="AI Engineer",
            role="AI Engineer",
            company="Marva AI",
            role_confidence=0.95,
            company_confidence=0.92,
        ),
    ]

    builder = CandidateBuilder()

    candidates = builder.build_candidates(
        anchors
    )

    assert len(candidates) == 2

    assert candidates[0].role == "Data Analyst"
    assert candidates[1].role == "AI Engineer"