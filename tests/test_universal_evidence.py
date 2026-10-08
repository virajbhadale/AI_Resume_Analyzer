from src.document_models import DocumentBlock
from src.inference.candidate_builder import (
    MultiBlockCandidateBuilder,
)
from src.inference.evidence_models import (
    EvidencePolarity,
    EvidenceType,
)


def make_block(
    block_id,
    text,
    *,
    section="experience",
    page=1,
    column_id=0,
    reading_order=1,
    bold=False,
    list_item=False,
    metadata=None,
):
    return DocumentBlock(
        id=block_id,
        text=text,
        page=page,
        column_id=column_id,
        reading_order=reading_order,
        is_bold=bold,
        is_list_item=list_item,
        section=section,
        metadata=metadata or {},
    )


def test_block_evidence_preserves_source_and_layout_provenance():

    block = make_block(
        "b1",
        "Data Analyst",
        page=2,
        column_id=1,
        reading_order=17,
        bold=True,
        metadata={
            "raw_text": " Data Analyst ",
            "normalized_text": "Data Analyst",
            "layout_region": "main_column",
            "section_confidence": 0.96,
        },
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.block_id == "b1"
    assert evidence.text == "Data Analyst"

    assert evidence.raw_text == " Data Analyst "
    assert evidence.normalized_text == "Data Analyst"

    assert evidence.page == 2
    assert evidence.column_id == 1
    assert evidence.reading_order == 17

    assert evidence.layout_region == "main_column"
    assert evidence.section == "experience"
    assert evidence.section_confidence == 0.96

    assert evidence.is_bold is True

    assert evidence.provenance["document_block_id"] == "b1"


def test_block_evidence_preserves_extraction_provenance():

    block = make_block(
        "b1",
        "AI Engineer",
    )

    block.source_type = "pdf"
    block.extraction_method = "pdf_text"

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.source_type == "pdf"
    assert evidence.extraction_method == "pdf_text"
    assert evidence.detector == "AnchorDetector"
    assert evidence.source_confidence == block.confidence

    assert evidence.provenance["source_type"] == "pdf"
    assert (
        evidence.provenance["extraction_method"]
        == "pdf_text"
    )


def test_list_item_creates_contradictory_structural_evidence():

    block = make_block(
        "b1",
        "- Developed machine learning models",
        list_item=True,
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    structural = [
        signal
        for signal in evidence.signals
        if signal.signal_type == EvidenceType.STRUCTURAL
    ]

    assert structural

    assert any(
        signal.value == "list_item"
        and signal.polarity == EvidencePolarity.CONTRADICT
        for signal in structural
    )


def test_sentence_like_block_creates_negative_semantic_evidence():

    block = make_block(
        "b1",
        "Developed scalable machine learning models for production.",
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    semantic = [
        signal
        for signal in evidence.signals
        if signal.signal_type == EvidenceType.SEMANTIC
    ]

    assert semantic

    assert any(
        signal.value == "sentence_like"
        and signal.polarity == EvidencePolarity.CONTRADICT
        for signal in semantic
    )


def test_section_context_is_preserved_without_becoming_final_decision():

    block = make_block(
        "b1",
        "Software Engineer",
        section="projects",
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.section == "projects"

    # BlockEvidence records context.
    # It does not itself reject the block.
    assert evidence.block_id == "b1"


def test_role_company_date_evidence_is_preserved():

    blocks = [
        make_block(
            "b1",
            "Machine Learning Engineer",
            bold=True,
            reading_order=1,
        ),
        make_block(
            "b2",
            "OpenAI",
            reading_order=2,
        ),
        make_block(
            "b3",
            "2024 - Present",
            reading_order=3,
        ),
    ]

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence(blocks)
    )

    assert len(evidence) == 3

    assert any(
        signal.signal_type == EvidenceType.ROLE
        for signal in evidence[0].signals
    )

    assert any(
        signal.signal_type == EvidenceType.COMPANY
        for signal in evidence[1].signals
    )

    assert any(
        signal.signal_type == EvidenceType.DATE
        for signal in evidence[2].signals
    )


def test_missing_metadata_is_preserved_as_missing():

    block = make_block(
        "b1",
        "Research Assistant",
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.role is not None

    assert evidence.company is None
    assert evidence.location is None
    assert evidence.start_date is None
    assert evidence.end_date is None


def test_block_evidence_does_not_convert_missing_fields_to_placeholders():

    block = make_block(
        "b1",
        "Research Assistant",
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.company is None
    assert evidence.location is None

    assert (
        "Company Not Specified"
        not in str(evidence.model_dump())
    )

def test_standalone_location_creates_location_evidence():

    block = make_block(
        "b1",
        "Pune, India",
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.location == "Pune, India"
    assert evidence.location_confidence > 0.0
    assert evidence.has_location is True


def test_us_location_creates_location_evidence():

    block = make_block(
        "b1",
        "San Francisco, CA",
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.location == "San Francisco, CA"
    assert evidence.location_confidence > 0.0
    assert evidence.has_location is True


def test_explicit_location_metadata_creates_location_evidence():

    block = make_block(
        "b1",
        "(Location - Pune, India)",
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.location == "Pune, India"
    assert evidence.location_confidence > 0.0
    assert evidence.has_location is True


def test_company_with_embedded_location_preserves_location_evidence():

    block = make_block(
        "b1",
        "Marva AI (Location - Pune, India)",
    )

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence([block])
    )[0]

    assert evidence.location == "Pune, India"
    assert evidence.location_confidence > 0.0
    assert evidence.has_location is True    