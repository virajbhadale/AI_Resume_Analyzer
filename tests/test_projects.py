"""
Tests for universal project extraction.

Run:
    pytest -q tests/test_projects.py
"""

from src.document_models import DocumentBlock
from src.extraction.projects import ProjectExtractor


def make_block(
    text: str,
    *,
    is_bold: bool = False,
    section: str = "projects",
) -> DocumentBlock:
    return DocumentBlock(
        id="test_project_1",
        text=text,
        section=section,
        is_heading=False,
        is_bold=is_bold,
    )


def extract(text: str, *, is_bold: bool = False, section: str = "projects"):
    extractor = ProjectExtractor()
    return extractor.extract(
        [make_block(text, is_bold=is_bold, section=section)]
    )


# ============================================================
# 1. BASIC PROJECT
# ============================================================

def test_basic_project_extraction():
    results = extract(
        """AI Resume Analyzer
Built an intelligent resume parsing system using Python and FastAPI."""
    )

    assert len(results) == 1

    project = results[0]

    assert project.title == "AI Resume Analyzer"
    assert "resume" in project.description.lower()


# ============================================================
# 2. PROJECT TECHNOLOGIES
# ============================================================

def test_project_extracts_technologies():
    results = extract(
        """AI Resume Analyzer
Built FastAPI services using Python and Docker."""
    )

    assert len(results) == 1

    technologies = results[0].technologies

    assert "Python" in technologies
    assert "FastAPI" in technologies
    assert "Docker" in technologies


# ============================================================
# 3. MULTIPLE PROJECTS
# ============================================================

def test_multiple_projects():
    blocks = [
        make_block(
            "AI Resume Analyzer\nBuilt a resume analysis system.",
            is_bold=True,
        ),
        make_block(
            "Crop Disease Detection\nDetected plant diseases using Python.",
            is_bold=True,
        ),
    ]

    results = ProjectExtractor().extract(blocks)

    assert len(results) == 2

    assert results[0].title == "AI Resume Analyzer"
    assert results[1].title == "Crop Disease Detection"


# ============================================================
# 4. SEMICOLON-SEPARATED PROJECTS
# ============================================================

def test_semicolon_separated_projects():
    results = extract(
        "AI Resume Analyzer; Crop Disease Detection; Travel AI Planner"
    )

    assert len(results) == 3

    titles = [project.title for project in results]

    assert "AI Resume Analyzer" in titles
    assert "Crop Disease Detection" in titles
    assert "Travel AI Planner" in titles


# ============================================================
# 5. PROJECT URL
# ============================================================

def test_project_url_is_extracted():
    results = extract(
        """AI Resume Analyzer
Built an intelligent resume analysis application.
https://github.com/example/ai-resume-analyzer"""
    )

    assert len(results) == 1

    project = results[0]

    assert project.link == "https://github.com/example/ai-resume-analyzer"


# ============================================================
# 6. BOLD PROJECT TITLE
# ============================================================

def test_bold_project_title():
    results = extract(
        "AI Farmer Crop Price Predictor\nPredicts agricultural commodity prices.",
        is_bold=True,
    )

    assert len(results) == 1

    assert results[0].title == "AI Farmer Crop Price Predictor"


# ============================================================
# 7. COLON PROJECT FORMAT
# ============================================================

def test_colon_project_format():
    results = extract(
        "AI Resume Analyzer: Intelligent PDF and DOCX resume parsing using Python."
    )

    assert len(results) == 1

    project = results[0]

    assert project.title == "AI Resume Analyzer"
    assert "Python" in project.technologies


# ============================================================
# 8. MULTI-LINE DESCRIPTION
# ============================================================

def test_multiline_project_description():
    results = extract(
        """Travel AI Planner
Built a multi-agent itinerary planning system.
Integrated weather and flight information.
Used LangGraph and Python."""
    )

    assert len(results) == 1

    project = results[0]

    assert "multi-agent" in project.description.lower()
    assert "weather" in project.description.lower()
    assert "LangGraph" in project.description


# ============================================================
# 9. PROJECT WITHOUT TECHNOLOGIES
# ============================================================

def test_project_without_technologies():
    results = extract(
        """Portfolio Website
Created a personal portfolio website and documented academic work."""
    )

    assert len(results) == 1

    project = results[0]

    assert project.title == "Portfolio Website"
    assert isinstance(project.technologies, list)


# ============================================================
# 10. EMPTY PROJECT SECTION
# ============================================================

def test_empty_project_input():
    results = ProjectExtractor().extract([])

    assert isinstance(results, list)
    assert len(results) == 0


# ============================================================
# 11. NON-PROJECT SECTIONS ARE IGNORED
# ============================================================

def test_non_project_sections_are_ignored():
    results = extract(
        """AI Resume Analyzer
Built a resume analysis system.""",
        section="experience",
    )

    assert results == []


# ============================================================
# 12. PROJECT TECHNOLOGIES FROM MULTIPLE LINES
# ============================================================

def test_technologies_are_collected_from_description():
    results = extract(
        """RAG Chatbot
Built a retrieval augmented generation system.
Used Python, LangChain, Docker, and FastAPI."""
    )

    assert len(results) == 1

    technologies = results[0].technologies

    assert "Python" in technologies
    assert "LangChain" in technologies
    assert "Docker" in technologies
    assert "FastAPI" in technologies


# ============================================================
# 13. GITHUB URL WITH TRAILING TEXT
# ============================================================

def test_project_url_with_trailing_text():
    results = extract(
        """AI Resume Analyzer
Built a resume analysis system.
GitHub: https://github.com/example/resume-analyzer"""
    )

    assert len(results) == 1

    assert results[0].link.startswith(
        "https://github.com/example/resume-analyzer"
    )


# ============================================================
# 14. PROJECT EXTRACTION NEVER CRASHES
# ============================================================

def test_unusual_project_text_does_not_crash():
    results = extract(
        """Project
AI/ML system — version 2.0
Technologies: Python, FastAPI, Docker
https://github.com/example/project"""
    )

    assert isinstance(results, list)

    if results:
        assert results[0].title
        assert isinstance(results[0].description, str)
        assert isinstance(results[0].technologies, list)