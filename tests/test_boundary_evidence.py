from src.inference.evidence_models import (
    BoundaryEvidence,
    BoundaryType,
    EvidencePolarity,
    EvidenceSignal,
    EvidenceType,
)


def test_boundary_defaults():

    boundary = BoundaryEvidence()

    assert boundary.boundary_type == BoundaryType.UNKNOWN
    assert boundary.score == 0.0

    assert boundary.previous_block_ids == []
    assert boundary.next_block_ids == []

    assert boundary.supporting_evidence == []
    assert boundary.contradictory_evidence == []


def test_new_role_boundary():

    boundary = BoundaryEvidence(
        boundary_type=BoundaryType.NEW_ROLE,
        score=0.91,
        previous_block_ids=["b10"],
        next_block_ids=["b11"],
    )

    assert boundary.boundary_type == BoundaryType.NEW_ROLE
    assert boundary.score == 0.91

    assert boundary.previous_block_ids == ["b10"]
    assert boundary.next_block_ids == ["b11"]


def test_block_deduplication():

    boundary = BoundaryEvidence()

    boundary.add_previous_block("b1")
    boundary.add_previous_block("b1")
    boundary.add_previous_block("b2")

    boundary.add_next_block("b3")
    boundary.add_next_block("b3")

    assert boundary.previous_block_ids == [
        "b1",
        "b2",
    ]

    assert boundary.next_block_ids == [
        "b3",
    ]


def test_supporting_evidence():

    boundary = BoundaryEvidence(
        boundary_type=BoundaryType.NEW_COMPANY,
        score=0.87,
    )

    signal = EvidenceSignal(
        signal_type=EvidenceType.COMPANY,
        value="Marva AI",
        confidence=0.94,
        polarity=EvidencePolarity.SUPPORT,
    )

    boundary.add_supporting_evidence(signal)

    assert len(boundary.supporting_evidence) == 1

    assert (
        boundary.supporting_evidence[0].value
        == "Marva AI"
    )


def test_contradictory_evidence():

    boundary = BoundaryEvidence(
        boundary_type=BoundaryType.NEW_ROLE,
        score=0.30,
    )

    signal = EvidenceSignal(
        signal_type=EvidenceType.STRUCTURAL,
        value="bullet continuation",
        confidence=0.91,
        polarity=EvidencePolarity.CONTRADICT,
    )

    boundary.add_contradictory_evidence(signal)

    assert len(boundary.contradictory_evidence) == 1

    assert (
        boundary.contradictory_evidence[0].polarity
        == EvidencePolarity.CONTRADICT
    )


def test_description_continuation():

    boundary = BoundaryEvidence(
        boundary_type=BoundaryType.DESCRIPTION_CONTINUATION,
        score=0.12,
        explanation=(
            "The next block appears to continue "
            "the previous experience description."
        ),
    )

    assert (
        boundary.boundary_type
        == BoundaryType.DESCRIPTION_CONTINUATION
    )

    assert boundary.score == 0.12

    assert "continue" in boundary.explanation