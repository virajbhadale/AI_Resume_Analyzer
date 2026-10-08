from src.inference.evidence_models import (
    EvidenceSignal,
    EvidenceType,
    EvidencePolarity,
)


def make_signal(
    signal_type,
    polarity,
    value,
    confidence,
    source_block_id,
    source_field=None,
    explanation=None,
    metadata=None,
):
    return EvidenceSignal(
        signal_type=signal_type,
        polarity=polarity,
        value=value,
        confidence=confidence,
        source_block_id=source_block_id,
        source_field=source_field,
        explanation=explanation,
        metadata=metadata or {},
    )


def test_evidence_signal_preserves_source_provenance():
    signal = make_signal(
        signal_type=EvidenceType.ROLE,
        polarity=EvidencePolarity.SUPPORT,
        value="Data Scientist",
        confidence=0.95,
        source_block_id="b1",
        source_field="role",
        explanation="Detected as a role anchor.",
    )

    assert signal.source_block_id == "b1"
    assert signal.source_field == "role"
    assert signal.value == "Data Scientist"
    assert signal.confidence == 0.95
    assert signal.explanation == "Detected as a role anchor."


def test_supporting_and_contradictory_evidence_are_distinguishable():
    supporting = make_signal(
        signal_type=EvidenceType.ROLE,
        polarity=EvidencePolarity.SUPPORT,
        value="Data Scientist",
        confidence=0.95,
        source_block_id="b1",
    )

    contradictory = make_signal(
        signal_type=EvidenceType.STRUCTURAL,
        polarity=EvidencePolarity.CONTRADICT,
        value="sentence_like",
        confidence=0.80,
        source_block_id="b2",
    )

    assert supporting.polarity != contradictory.polarity
    assert supporting.source_block_id != contradictory.source_block_id


def test_evidence_confidence_is_bounded():
    signal = make_signal(
        signal_type=EvidenceType.ROLE,
        polarity=EvidencePolarity.SUPPORT,
        value="Data Scientist",
        confidence=0.75,
        source_block_id="b1",
    )

    assert 0.0 <= signal.confidence <= 1.0


def test_evidence_preserves_source_field():
    signal = make_signal(
        signal_type=EvidenceType.COMPANY,
        polarity=EvidencePolarity.SUPPORT,
        value="ABC Technologies",
        confidence=0.90,
        source_block_id="b2",
        source_field="company",
    )

    assert signal.source_field == "company"


def test_evidence_source_block_is_not_lost():
    signal = make_signal(
        signal_type=EvidenceType.DATE,
        polarity=EvidencePolarity.SUPPORT,
        value="2023 - 2024",
        confidence=0.90,
        source_block_id="b3",
    )

    assert signal.source_block_id == "b3"


def test_multiple_evidence_signals_can_reference_different_blocks():
    signals = [
        make_signal(
            signal_type=EvidenceType.ROLE,
            polarity=EvidencePolarity.SUPPORT,
            value="Data Scientist",
            confidence=0.95,
            source_block_id="b1",
        ),
        make_signal(
            signal_type=EvidenceType.COMPANY,
            polarity=EvidencePolarity.SUPPORT,
            value="ABC Technologies",
            confidence=0.90,
            source_block_id="b2",
        ),
        make_signal(
            signal_type=EvidenceType.DATE,
            polarity=EvidencePolarity.SUPPORT,
            value="2023 - 2024",
            confidence=0.90,
            source_block_id="b3",
        ),
    ]

    assert [signal.source_block_id for signal in signals] == [
        "b1",
        "b2",
        "b3",
    ]


def test_weak_evidence_does_not_become_high_confidence():
    signal = make_signal(
        signal_type=EvidenceType.ROLE,
        polarity=EvidencePolarity.SUPPORT,
        value="Engineer",
        confidence=0.45,
        source_block_id="b1",
    )

    assert signal.confidence < 0.80


def test_contradictory_evidence_remains_explicit():
    signal = make_signal(
        signal_type=EvidenceType.STRUCTURAL,
        polarity=EvidencePolarity.CONTRADICT,
        value="sentence_like",
        confidence=0.80,
        source_block_id="b4",
        explanation="Sentence-like prose is negative evidence for an anchor.",
    )

    assert signal.polarity == EvidencePolarity.CONTRADICT
    assert signal.confidence == 0.80
    assert signal.explanation is not None


def test_evidence_order_is_deterministic():
    signals = [
        make_signal(
            signal_type=EvidenceType.ROLE,
            polarity=EvidencePolarity.SUPPORT,
            value="Data Scientist",
            confidence=0.95,
            source_block_id="b1",
        ),
        make_signal(
            signal_type=EvidenceType.COMPANY,
            polarity=EvidencePolarity.SUPPORT,
            value="ABC Technologies",
            confidence=0.90,
            source_block_id="b2",
        ),
    ]

    first = [
        (s.signal_type, s.source_block_id, s.confidence)
        for s in signals
    ]

    second = [
        (s.signal_type, s.source_block_id, s.confidence)
        for s in signals
    ]

    assert first == second


def test_provenance_contains_enough_information_to_explain_signal():
    signal = make_signal(
        signal_type=EvidenceType.ROLE,
        polarity=EvidencePolarity.SUPPORT,
        value="Machine Learning Engineer",
        confidence=0.95,
        source_block_id="experience-block-7",
        source_field="role",
        explanation="Detected from a role anchor.",
    )

    explanation = {
        "source_block_id": signal.source_block_id,
        "source_field": signal.source_field,
        "value": signal.value,
        "confidence": signal.confidence,
        "polarity": signal.polarity,
        "signal_type": signal.signal_type,
        "explanation": signal.explanation,
    }

    assert explanation["source_block_id"] == "experience-block-7"
    assert explanation["source_field"] == "role"
    assert explanation["value"] == "Machine Learning Engineer"
    assert explanation["confidence"] == 0.95
    assert explanation["explanation"] == "Detected from a role anchor."