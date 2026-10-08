"""
Universal section-detection behavior tests.

These tests intentionally avoid resume-template-specific coordinates
or assumptions.
"""

from src.document_models import BBox, DocumentBlock
from src.section_extractor import SectionExtractor


def make_block(
    block_id,
    text,
    y0,
    y1,
    *,
    bold=False,
    font_size=10.0,
    column_id=0,
    page=1,
):
    return DocumentBlock(
        id=block_id,
        text=text,
        page=page,
        page_width=600,
        page_height=800,
        bbox=BBox(
            x0=20,
            y0=y0,
            x1=580,
            y1=y1,
        ),
        font_size=font_size,
        is_bold=bold,
        column_id=column_id,
    )


def test_exact_section_headings_are_detected():

    extractor = SectionExtractor()

    result = extractor.classify_heading(
        "WORK EXPERIENCE",
        is_bold=True,
        font_size=14,
    )

    assert result is not None
    assert result[0] == "experience"


def test_sentence_is_not_a_section_heading():

    extractor = SectionExtractor()

    result = extractor.classify_heading(
        "Developed scalable machine learning systems.",
        is_bold=True,
        font_size=14,
    )

    assert result is None


def test_heading_does_not_require_bold_when_configured_exactly():

    extractor = SectionExtractor()

    result = extractor.classify_heading(
        "EDUCATION",
        is_bold=False,
        font_size=10,
    )

    assert result is not None
    assert result[0] == "education"


def test_technical_punctuation_is_not_destroyed_by_heading_normalization():

    extractor = SectionExtractor()

    original = "C++ / C# / .NET"

    normalized = extractor._normalize_heading(
        original
    )

    assert "c++" in normalized
    assert "c#" in normalized
    assert ".net" in normalized


def test_section_transition_is_preserved():

    extractor = SectionExtractor()

    blocks = [
        make_block(
            "b1",
            "SKILLS",
            20,
            40,
            bold=True,
            font_size=14,
        ),
        make_block(
            "b2",
            "Python, SQL, PyTorch",
            50,
            70,
        ),
        make_block(
            "b3",
            "EXPERIENCE",
            100,
            120,
            bold=True,
            font_size=14,
        ),
        make_block(
            "b4",
            "Machine Learning Engineer",
            130,
            150,
        ),
    ]

    tagged = (
        extractor.segment_document_by_sections(
            blocks
        )
    )

    assert tagged[0].section == "skills"
    assert tagged[1].section == "skills"
    assert tagged[2].section == "experience"
    assert tagged[3].section == "experience"


def test_multiple_columns_keep_section_detection_local():

    extractor = SectionExtractor()

    blocks = [
        make_block(
            "left_heading",
            "EXPERIENCE",
            20,
            40,
            bold=True,
            font_size=14,
            column_id=0,
        ),
        make_block(
            "left_content",
            "Software Engineer",
            50,
            70,
            column_id=0,
        ),
        make_block(
            "right_heading",
            "EDUCATION",
            20,
            40,
            bold=True,
            font_size=14,
            column_id=1,
        ),
        make_block(
            "right_content",
            "Master of Science",
            50,
            70,
            column_id=1,
        ),
    ]

    tagged = (
        extractor.segment_document_by_sections(
            blocks
        )
    )

    assert tagged[0].section == "experience"
    assert tagged[1].section == "experience"

    assert tagged[2].section == "education"
    assert tagged[3].section == "education"