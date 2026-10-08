"""Integration tests for the complete AI Resume Analyzer V2 pipeline."""

import os

from src.pipeline import ResumeAnalysisPipeline, analyze_resume


SAMPLE_RESUME = "data/samples/sample_resume_1.txt"


def test_pipeline_fixture_exists():
    """The primary integration-test resume fixture must exist."""
    assert os.path.exists(SAMPLE_RESUME)


def test_pipeline_returns_complete_result():
    """The complete pipeline should return all major output sections."""
    result = analyze_resume(SAMPLE_RESUME)

    assert isinstance(result, dict)

    assert "resume" in result
    assert "ats_analysis" in result
    assert "confidence" in result
    assert "episode_graph" in result
    assert "jd_match" in result
    assert "layout_summary" in result


def test_pipeline_resume_structure():
    """The canonical resume should contain all major resume sections."""
    result = analyze_resume(SAMPLE_RESUME)

    resume = result["resume"]

    assert isinstance(resume, dict)

    assert "contact" in resume
    assert "current_title" in resume
    assert "domain_category" in resume
    assert "summary" in resume
    assert "skills" in resume
    assert "experience" in resume
    assert "education" in resume
    assert "projects" in resume
    assert "certifications" in resume
    assert "publications" in resume
    assert "awards" in resume
    assert "achievements" in resume
    assert "languages" in resume
    assert "interests" in resume
    assert "volunteer_experience" in resume
    assert "references" in resume
    assert "metadata" in resume


def test_pipeline_metadata():
    """Pipeline metadata should identify the processed source."""
    result = analyze_resume(SAMPLE_RESUME)

    metadata = result["resume"]["metadata"]

    assert metadata["source_file"] == "sample_resume_1.txt"
    assert metadata["total_blocks_processed"] > 0
    assert isinstance(metadata["sections_detected"], list)


def test_pipeline_layout_summary():
    """Layout summary should contain basic processing information."""
    result = analyze_resume(SAMPLE_RESUME)

    layout_summary = result["layout_summary"]

    assert "total_blocks" in layout_summary
    assert "columns_detected" in layout_summary
    assert "sections_found" in layout_summary

    assert layout_summary["total_blocks"] > 0
    assert layout_summary["columns_detected"] >= 1


def test_pipeline_extracted_entities_are_lists():
    """Collection-based resume entities should have list values."""
    result = analyze_resume(SAMPLE_RESUME)

    resume = result["resume"]

    assert isinstance(resume["skills"], list)
    assert isinstance(resume["experience"], list)
    assert isinstance(resume["education"], list)
    assert isinstance(resume["projects"], list)
    assert isinstance(resume["certifications"], list)
    assert isinstance(resume["publications"], list)
    assert isinstance(resume["awards"], list)
    assert isinstance(resume["achievements"], list)
    assert isinstance(resume["languages"], list)
    assert isinstance(resume["interests"], list)
    assert isinstance(resume["volunteer_experience"], list)
    assert isinstance(resume["references"], list)


def test_pipeline_with_job_description():
    """JD matching should be activated when valid JD text is supplied."""
    jd_text = """
    We are looking for a Data Scientist with experience in Python,
    machine learning, SQL, pandas, scikit-learn, and data analysis.
    """

    result = analyze_resume(
        SAMPLE_RESUME,
        target_role="data_scientist",
        jd_text=jd_text,
    )

    assert result["jd_match"] is not None

    jd_match = result["jd_match"]

    assert isinstance(jd_match, dict)
    assert "similarity_score" in jd_match


def test_pipeline_without_job_description():
    """JD matching should remain disabled when no JD is supplied."""
    result = analyze_resume(SAMPLE_RESUME)

    assert result["jd_match"] is None


def test_class_pipeline_api():
    """The class-based pipeline API should work independently."""
    pipeline = ResumeAnalysisPipeline()

    result = pipeline.analyze_resume(
        SAMPLE_RESUME,
        target_role="data_scientist",
    )

    assert isinstance(result, dict)
    assert "resume" in result
    assert "ats_analysis" in result
    assert "confidence" in result
    assert "episode_graph" in result
    assert "layout_summary" in result