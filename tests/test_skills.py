"""Tests for Technical Skills Extraction."""
import pytest
from src.extraction.skills import SkillExtractor
from src.document_models import DocumentBlock

def test_skills_extraction():
    extractor = SkillExtractor()
    sample_text = (
        "Experienced in Python, SQL, and PyTorch. "
        "Built automated RAG applications using LangChain and Pinecone vector databases. "
        "Containerized services with Docker and orchestrated CI/CD workflows in Git."
    )

    skills = extractor.extract_from_text(sample_text)
    assert "Python" in skills
    assert "SQL" in skills
    assert "PyTorch" in skills
    assert "RAG" in skills
    assert "LangChain" in skills
    assert "Pinecone" in skills
    assert "Docker" in skills
    assert "Git" in skills

def test_skills_section_priority():
    extractor = SkillExtractor()
    blocks = [
        DocumentBlock(id="b1", text="SKILLS", section="skills", is_heading=True),
        DocumentBlock(id="b2", text="Languages: Python, SQL, C++\nFrameworks: Scikit-Learn, TensorFlow", section="skills")
    ]

    skills = extractor.extract(blocks)
    assert "Python" in skills
    assert "Scikit-Learn" in skills
    assert "TensorFlow" in skills
