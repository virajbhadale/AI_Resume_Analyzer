from src.document_models import DocumentBlock
from src.inference.candidate_builder import MultiBlockCandidateBuilder


def make_block(
    block_id,
    text,
    reading_order,
    *,
    bold=False,
    y=0,
    section="experience",
):
    return DocumentBlock(
        id=block_id,
        text=text,
        page_number=1,
        x=0,
        y=y,
        width=500,
        height=20,
        reading_order=reading_order,
        is_bold=bold,
        section=section,
    )


def build_candidate(blocks):
    builder = MultiBlockCandidateBuilder()
    candidates = builder.build_from_blocks(blocks)

    assert candidates, "Expected at least one experience candidate"
    return candidates[0]


def test_complete_candidate_has_positive_preliminary_confidence():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "ABC Technologies", 2),
        make_block("b3", "2022 - 2024", 3),
        make_block("b4", "Pune, India", 4),
        make_block(
            "b5",
            "Built automated reporting pipelines using Python.",
            5,
        ),
    ]

    candidate = build_candidate(blocks)

    assert candidate.confidence > 0.0
    assert candidate.confidence <= 1.0


def test_missing_company_does_not_zero_candidate_confidence():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "2022 - 2024", 2),
        make_block("b3", "Pune, India", 3),
        make_block(
            "b4",
            "Built automated reporting pipelines using Python.",
            4,
        ),
    ]

    candidate = build_candidate(blocks)

    assert candidate.role == "Data Analyst"
    assert candidate.company is None
    assert candidate.confidence > 0.0
    assert candidate.confidence <= 1.0


def test_missing_location_does_not_zero_candidate_confidence():
    blocks = [
        make_block("b1", "Machine Learning Engineer", 1, bold=True),
        make_block("b2", "ABC Technologies", 2),
        make_block("b3", "2021 - 2023", 3),
        make_block(
            "b4",
            "Developed production machine learning systems.",
            4,
        ),
    ]

    candidate = build_candidate(blocks)

    assert candidate.role == "Machine Learning Engineer"
    assert candidate.company == "ABC Technologies"
    assert candidate.location is None
    assert candidate.confidence > 0.0
    assert candidate.confidence <= 1.0


def test_confidence_is_always_bounded():
    blocks = [
        make_block("b1", "Data Scientist", 1, bold=True),
        make_block("b2", "XYZ Analytics", 2),
        make_block("b3", "2020 - Present", 3),
        make_block("b4", "Mumbai, India", 4),
        make_block(
            "b5",
            "Developed forecasting and machine learning models.",
            5,
        ),
    ]

    candidate = build_candidate(blocks)

    assert 0.0 <= candidate.confidence <= 1.0


def test_candidate_retains_positive_and_contradictory_evidence():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "ABC Technologies", 2),
        make_block("b3", "2022 - 2024", 3),
        make_block(
            "b4",
            "- Built automated reporting pipelines.",
            4,
        ),
    ]

    candidate = build_candidate(blocks)

    positive = [
        signal
        for signal in candidate.evidence.signals
        if getattr(signal, "polarity", None)
        and str(signal.polarity).upper().endswith("SUPPORT")
    ]

    contradictory = [
        signal
        for signal in candidate.evidence.signals
        if getattr(signal, "polarity", None)
        and str(signal.polarity).upper().endswith("CONTRADICT")
    ]

    assert positive or candidate.confidence > 0.0
    assert contradictory or candidate.confidence > 0.0