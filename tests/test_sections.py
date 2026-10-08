"""Tests for Section Detection and Canonical Mapping."""
import pytest
from src.section_extractor import SectionExtractor
from src.document_models import DocumentBlock

def test_heading_classification():
    extractor = SectionExtractor()

    # Exact match
    res = extractor.classify_heading("WORK EXPERIENCE", is_bold=True, font_size=14.0)
    assert res is not None
    sec, score = res
    assert sec == "experience"
    assert score >= 0.85

    # Synonyms
    res_edu = extractor.classify_heading("Academic Background", is_bold=True)
    assert res_edu is not None
    assert res_edu[0] == "education"

    res_skills = extractor.classify_heading("Technical Proficiencies", is_bold=True)
    assert res_skills is not None
    assert res_skills[0] == "skills"

    # Non-heading text should return None
    res_non = extractor.classify_heading("Developed scalable machine learning models in Python for production.", is_bold=False)
    assert res_non is None

def test_document_segmentation():
    extractor = SectionExtractor()
    blocks = [
        DocumentBlock(id="b1", text="Jane Doe\njane@example.com", font_size=16.0),
        DocumentBlock(id="b2", text="SKILLS", is_bold=True, font_size=14.0),
        DocumentBlock(id="b3", text="Python, SQL, PyTorch"),
        DocumentBlock(id="b4", text="EXPERIENCE", is_bold=True, font_size=14.0),
        DocumentBlock(id="b5", text="Data Scientist at ACME Corp\n2022 - Present")
    ]

    tagged = extractor.segment_document_by_sections(blocks)
    assert tagged[0].section == "contact"
    assert tagged[1].section == "skills"
    assert tagged[2].section == "skills"
    assert tagged[3].section == "experience"
    assert tagged[4].section == "experience"

def test_heading_classification_with_decorative_underlines():
    extractor = SectionExtractor()

    cases = [
        ("Experience ______________________________", "experience"),
        ("WORK EXPERIENCE __________________________", "experience"),
        ("Professional Summary ____________________", "summary"),
        ("Technical Skills ________________________", "skills"),
    ]

    for heading, expected_section in cases:
        result = extractor.classify_heading(
            heading,
            is_bold=True,
            font_size=12.0,
        )

        assert result is not None, heading
        assert result[0] == expected_section

def test_experience_heading_with_long_decorative_rule():
    extractor = SectionExtractor()

    heading = (
        "Experience "
        "___________________________________________________________________________________________"
    )

    result = extractor.classify_heading(
        heading,
        is_bold=True,
        font_size=12.0,
    )

    assert result is not None
    assert result[0] == "experience"
    
            