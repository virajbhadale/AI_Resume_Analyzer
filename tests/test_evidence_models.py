from src.inference.evidence_models import (
    EvidenceCollection,
    EvidenceHypothesis,
    EvidencePolarity,
    EvidenceResolution,
    EvidenceSignal,
    EvidenceType,
    FieldEvidence
)


def test_evidence_signal_defCaults():

    signal = EvidenceSignal(
        signal_type=EvidenceType.ROLE,
        value="Data Scientist",
        confidence=0.92,
    )

    assert signal.signal_type == EvidenceType.ROLE
    assert signal.value == "Data Scientist"
    assert signal.confidence == 0.92
    assert signal.polarity == EvidencePolarity.SUPPORT


def test_evidence_signal_supporting_explanation():

    signal = EvidenceSignal(
        signal_type=EvidenceType.TYPOGRAPHY,
        value="bold",
        confidence=0.85,
        explanation="Role candidate is visually emphasized.",
    )

    assert signal.explanation == (
        "Role candidate is visually emphasized."
    )


def test_evidence_signal_confidence_validation():

    try:
        EvidenceSignal(
            signal_type=EvidenceType.ROLE,
            confidence=1.5,
        )
    except ValueError:
        return

    raise AssertionError(
        "Confidence above 1.0 should be rejected."
    )


def test_evidence_hypothesis():

    hypothesis = EvidenceHypothesis(
        label="ROLE",
        value="Machine Learning Engineer",
        confidence=0.88,
    )

    assert hypothesis.label == "ROLE"
    assert hypothesis.value == (
        "Machine Learning Engineer"
    )
    assert hypothesis.confidence == 0.88
    assert hypothesis.evidence == []


def test_hypothesis_add_evidence():

    hypothesis = EvidenceHypothesis(
        label="ROLE",
        value="Data Analyst",
    )

    signal = EvidenceSignal(
        signal_type=EvidenceType.ROLE,
        value="Data Analyst",
        confidence=0.90,
    )

    hypothesis.add_evidence(signal)

    assert len(hypothesis.evidence) == 1
    assert hypothesis.evidence[0].value == (
        "Data Analyst"
    )


def test_evidence_collection():

    collection = EvidenceCollection()

    role_signal = EvidenceSignal(
        signal_type=EvidenceType.ROLE,
        value="Data Engineer",
        confidence=0.91,
    )

    company_signal = EvidenceSignal(
        signal_type=EvidenceType.COMPANY,
        value="Example Corp",
        confidence=0.94,
    )

    collection.add_signal(role_signal)
    collection.add_signal(company_signal)

    assert len(collection.signals) == 2


def test_evidence_collection_filter():

    collection = EvidenceCollection()

    collection.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.ROLE,
            value="ML Engineer",
            confidence=0.90,
        )
    )

    collection.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.COMPANY,
            value="Example AI",
            confidence=0.95,
        )
    )

    roles = collection.signals_for(
        EvidenceType.ROLE
    )

    assert len(roles) == 1
    assert roles[0].value == "ML Engineer"


def test_resolution():

    signal = EvidenceSignal(
        signal_type=EvidenceType.DATE,
        value="Jan 2024 - Present",
        confidence=0.96,
    )

    resolution = EvidenceResolution(
        label="DATE",
        value="Jan 2024 - Present",
        confidence=0.96,
        accepted=True,
        evidence=[signal],
    )

    assert resolution.accepted is True
    assert resolution.confidence == 0.96
    assert len(resolution.evidence) == 1


def test_contradictory_evidence():

    signal = EvidenceSignal(
        signal_type=EvidenceType.STRUCTURAL,
        value="bullet",
        confidence=0.80,
        polarity=EvidencePolarity.CONTRADICT,
    )

    assert signal.polarity == (
        EvidencePolarity.CONTRADICT
    )

def test_field_evidence_defaults():

    evidence = FieldEvidence(
        field="role",
        value="Data Scientist",
        confidence=0.92,
    )

    assert evidence.field == "role"
    assert evidence.value == "Data Scientist"
    assert evidence.confidence == 0.92
    assert evidence.signals == []
    assert evidence.source_block_ids == []
    assert evidence.accepted is False


def test_field_evidence_add_signal():

    evidence = FieldEvidence(
        field="role",
        value="AI Engineer",
    )

    signal = EvidenceSignal(
        signal_type=EvidenceType.ROLE,
        value="AI Engineer",
        confidence=0.94,
        source_block_id="b17",
    )

    evidence.add_signal(signal)

    assert len(evidence.signals) == 1
    assert evidence.signals[0] == signal
    assert evidence.source_block_ids == ["b17"]

def test_field_evidence_multiple_signals_same_block():

    evidence = FieldEvidence(
        field="role",
        value="Machine Learning Engineer",
    )

    evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.ROLE,
            value="Machine Learning Engineer",
            confidence=0.94,
            source_block_id="b17",
        )
    )

    evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.TYPOGRAPHY,
            value="bold",
            confidence=0.85,
            source_block_id="b17",
        )
    )

    assert len(evidence.signals) == 2

    assert evidence.source_block_ids == ["b17"]

def test_field_evidence_multiple_source_blocks():

    evidence = FieldEvidence(
        field="company",
        value="Marva AI",
    )

    evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.COMPANY,
            value="Marva AI",
            confidence=0.91,
            source_block_id="b17",
        )
    )

    evidence.add_signal(
        EvidenceSignal(
            signal_type=EvidenceType.LAYOUT,
            value="header_position",
            confidence=0.82,
            source_block_id="b18",
        )
    )

    assert evidence.source_block_ids == [
        "b17",
        "b18",
    ]

def test_field_evidence_deduplicates_source_blocks():

    evidence = FieldEvidence(
        field="role",
        value="AI Engineer",
    )

    signal_1 = EvidenceSignal(
        signal_type=EvidenceType.ROLE,
        value="AI Engineer",
        confidence=0.94,
        source_block_id="b17",
    )

    signal_2 = EvidenceSignal(
        signal_type=EvidenceType.TYPOGRAPHY,
        value="bold",
        confidence=0.85,
        source_block_id="b17",
    )

    evidence.add_signal(signal_1)
    evidence.add_signal(signal_2)

    assert evidence.source_block_ids == ["b17"]

def test_field_evidence_confidence_validation():

    try:
        FieldEvidence(
            field="role",
            value="Engineer",
            confidence=1.5,
        )
    except ValueError:
        return

    raise AssertionError(
        "Confidence above 1.0 should be rejected."
    )

def test_field_evidence_negative_confidence():

    try:
        FieldEvidence(
            field="role",
            value="Engineer",
            confidence=-0.1,
        )
    except ValueError:
        return

    raise AssertionError(
        "Negative confidence should be rejected."
    )                        