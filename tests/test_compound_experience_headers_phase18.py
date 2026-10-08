"""
Phase 18B.5.9
Compound Experience Header Regression Tests

Purpose
-------
Protect against regression where a single document block contains
both a compound role and an organization name without an explicit
ROLE - COMPANY delimiter.

These tests intentionally target structural patterns observed during
Jillani resume forensic analysis.

Production code should extract evidence from the structure of the
header rather than relying on resume-specific names.
"""

from src.inference.anchors import AnchorDetector


def _analyze(text: str):
    detector = AnchorDetector()
    return detector.analyze_block(text)


def test_compound_role_and_company_without_delimiter():
    result = _analyze(
        "Senior Data Scientist & Machine Learning "
        "Software Engineer (Generative AI) PURELOGICS"
    )

    assert result.role is not None
    assert result.company is not None
    assert result.role != result.company


def test_compound_role_and_company_nvidia_pattern():
    result = _analyze(
        "Senior Data Scientist Machine Learning & "
        "Gen AI Engineer (Volunteer) NVIDIA"
    )

    assert result.role is not None
    assert result.company is not None
    assert result.role != result.company


def test_compound_role_and_company_blocbelt_pattern():
    result = _analyze(
        "Senior Data Scientist & Machine Learning Engineer BLOCBELT"
    )

    assert result.role is not None
    assert result.company is not None
    assert result.role != result.company


def test_compound_role_with_specialization_and_company():
    result = _analyze(
        "AI Data Scientist Kaggle Master Mentor "
        "(Part-Time, 20 hours/week)"
    )

    # This pattern contains a role-like header but does not
    # provide sufficiently explicit organization evidence.
    # The detector must not invent a company.
    assert result.role is not None
    assert result.company is None


def test_simple_role_header_remains_supported():
    result = _analyze("Data Scientist")

    assert result.role == "Data Scientist"
    assert result.company is None


def test_simple_company_header_remains_supported():
    result = _analyze("Crypto-Express")

    assert result.company == "Crypto-Express"
    assert result.role is None


def test_role_at_company_remains_supported():
    result = _analyze("Machine Learning Engineer @ OpenAI")

    assert result.role is not None
    assert result.company == "OpenAI"


def test_prose_must_not_become_compound_company():
    result = _analyze(
        "Presented findings at weekly research meetings"
    )

    assert result.company is None