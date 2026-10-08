from src.inference.evidence_models import (
    AnchorEvidence,
    AnchorStrength,
    AnchorType,
    EvidencePolarity,
    EvidenceSignal,
    EvidenceType,
)


def test_anchor_defaults():

    anchor = AnchorEvidence()

    assert anchor.anchor_type == AnchorType.UNKNOWN
    assert anchor.strength == AnchorStrength.WEAK
    assert anchor.score == 0.0
    assert anchor.block_ids == []
    assert anchor.supporting_evidence == []
    assert anchor.contradictory_evidence == []


def test_strong_anchor():

    anchor = AnchorEvidence(
        anchor_type=AnchorType.ROLE_COMPANY_DATE,
        strength=AnchorStrength.STRONG,
        score=0.94,
        block_ids=["b1", "b2", "b3"],
    )

    assert anchor.anchor_type == AnchorType.ROLE_COMPANY_DATE
    assert anchor.strength == AnchorStrength.STRONG
    assert anchor.score == 0.94
    assert anchor.is_strong is True
    assert anchor.is_medium is False
    assert anchor.is_weak is False


def test_anchor_block_deduplication():

    anchor = AnchorEvidence()

    anchor.add_block("b1")
    anchor.add_block("b2")
    anchor.add_block("b1")

    assert anchor.block_ids == ["b1", "b2"]


def test_supporting_evidence():

    anchor = AnchorEvidence(
        anchor_type=AnchorType.ROLE_DATE,
        strength=AnchorStrength.STRONG,
        score=0.90,
    )

    signal = EvidenceSignal(
        signal_type=EvidenceType.ROLE,
        value="AI Engineer",
        confidence=0.95,
        polarity=EvidencePolarity.SUPPORT,
    )

    anchor.add_supporting_evidence(signal)

    assert len(anchor.supporting_evidence) == 1
    assert anchor.supporting_evidence[0].value == "AI Engineer"


def test_contradictory_evidence():

    anchor = AnchorEvidence(
        anchor_type=AnchorType.ROLE_ONLY,
        strength=AnchorStrength.WEAK,
        score=0.40,
    )

    signal = EvidenceSignal(
        signal_type=EvidenceType.SECTION,
        value="skills",
        confidence=0.92,
        polarity=EvidencePolarity.CONTRADICT,
    )

    anchor.add_contradictory_evidence(signal)

    assert len(anchor.contradictory_evidence) == 1
    assert (
        anchor.contradictory_evidence[0].polarity
        == EvidencePolarity.CONTRADICT
    )


def test_anchor_strength_helpers():

    strong = AnchorEvidence(
        strength=AnchorStrength.STRONG
    )

    medium = AnchorEvidence(
        strength=AnchorStrength.MEDIUM
    )

    weak = AnchorEvidence(
        strength=AnchorStrength.WEAK
    )

    assert strong.is_strong is True
    assert medium.is_medium is True
    assert weak.is_weak is True


def test_anchor_explanation():

    anchor = AnchorEvidence(
        anchor_type=AnchorType.COMPANY_DATE_LOCATION,
        strength=AnchorStrength.MEDIUM,
        score=0.72,
        explanation=(
            "Company, date, and location evidence "
            "occur within the same experience region."
        ),
    )

    assert anchor.explanation is not None
    assert "Company" in anchor.explanation