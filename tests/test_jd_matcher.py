"""Tests for Job Description Matching Engine."""

from src.ats.jd_matcher import JobDescriptionMatcher
from src.document_models import (
    CanonicalResume,
    ContactInfo,
)


def build_resume() -> CanonicalResume:
    return CanonicalResume(
        contact=ContactInfo(
            name="Test User",
            email="test@example.com",
        ),
        summary=(
            "Data Scientist experienced in Python, "
            "machine learning, SQL, and pandas."
        ),
        skills=[
            "Python",
            "Machine Learning",
            "SQL",
            "Pandas",
        ],
    )


def test_jd_match_finds_matched_and_missing_skills():
    resume = build_resume()

    jd = """
    We are looking for a Data Scientist with experience in
    Python, machine learning, SQL, pandas, Docker, and Kubernetes.
    """

    matcher = JobDescriptionMatcher()
    result = matcher.match(resume, jd)

    assert isinstance(result, dict)

    assert "similarity_score" in result
    assert "matched_skills" in result
    assert "missing_skills" in result

    assert "Python" in result["matched_skills"]
    assert "Machine Learning" in result["matched_skills"]
    assert "SQL" in result["matched_skills"]
    assert "Pandas" in result["matched_skills"]

    assert "Docker" in result["missing_skills"]
    assert "Kubernetes" in result["missing_skills"]


def test_jd_match_is_case_insensitive():
    resume = build_resume()

    jd = """
    We need someone with PYTHON, SQL, MACHINE LEARNING,
    and PANDAS experience.
    """

    matcher = JobDescriptionMatcher()
    result = matcher.match(resume, jd)

    assert "Python" in result["matched_skills"]
    assert "SQL" in result["matched_skills"]
    assert "Machine Learning" in result["matched_skills"]
    assert "Pandas" in result["matched_skills"]


def test_jd_aliases_are_normalized():
    resume = build_resume()

    jd = """
    Required skills include Python, sklearn, and
    large language models.
    """

    matcher = JobDescriptionMatcher()
    result = matcher.match(resume, jd)

    assert "Python" in result["matched_skills"]
    assert "Scikit-Learn" in result["missing_skills"]
    assert "LLMs" in result["missing_skills"]


def test_invalid_job_description_returns_safe_result():
    resume = build_resume()

    matcher = JobDescriptionMatcher()

    result = matcher.match(resume, "too short")

    assert result["similarity_score"] == 0.0
    assert result["matched_skills"] == []
    assert result["missing_skills"] == []
    assert "valid Job Description" in result["key_highlights"]


def test_empty_job_description_returns_safe_result():
    resume = build_resume()

    matcher = JobDescriptionMatcher()

    result = matcher.match(resume, "")

    assert result["similarity_score"] == 0.0
    assert result["matched_skills"] == []
    assert result["missing_skills"] == []


def test_jd_result_contains_skill_count():
    resume = build_resume()

    jd = """
    Data Scientist required with Python, SQL, pandas,
    Docker and Kubernetes.
    """

    matcher = JobDescriptionMatcher()
    result = matcher.match(resume, jd)

    assert "total_jd_skills_found" in result
    assert result["total_jd_skills_found"] >= 5