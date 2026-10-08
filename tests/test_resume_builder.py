"""Tests for the canonical ResumeBuilder."""

from src.document_models import (
    ContactInfo,
    ExperienceEpisode,
    EducationEntry,
)
from src.inference.resume_builder import ResumeBuilder


def test_build_minimal_resume():

    builder = ResumeBuilder()

    resume = builder.build()

    assert resume.current_title == "Professional"
    assert resume.domain_category == "General"
    assert resume.summary == ""

    assert resume.skills == []
    assert resume.experience == []
    assert resume.education == []
    assert resume.projects == []


def test_build_complete_resume():

    builder = ResumeBuilder()

    contact = ContactInfo(
        name="Alex Chen",
        email="alex@example.com",
        location="Pune, Maharashtra",
    )

    experience = [
        ExperienceEpisode(
            id="exp-1",
            role="Data Scientist",
            company="Nexus Analytics",
            start_date="Jan 2023",
            end_date="Present",
            is_current=True,
            description="Built machine learning systems.",
        )
    ]

    education = [
        EducationEntry(
            id="edu-1",
            degree="B.Tech Computer Science",
            institution="MIT",
            start_date="2020",
            end_date="2024",
        )
    ]

    resume = builder.build(
        contact=contact,
        current_title="Data Scientist",
        domain_category="Data Science",
        summary="Data scientist with machine learning experience.",
        skills=["Python", "SQL", "Pandas"],
        experience=experience,
        education=education,
        projects=["Resume Analyzer"],
        certifications=["AWS Certified Cloud Practitioner"],
        publications=["Research Paper"],
        awards=["Best Project"],
        achievements=["Top Performer"],
        languages=["English"],
        interests=["AI Research"],
        volunteer_experience=["Teaching"],
        references=["Available on request"],
    )

    assert resume.contact.name == "Alex Chen"
    assert resume.current_title == "Data Scientist"
    assert resume.domain_category == "Data Science"

    assert "Python" in resume.skills
    assert len(resume.experience) == 1
    assert resume.experience[0].company == "Nexus Analytics"

    assert len(resume.education) == 1
    assert resume.education[0].institution == "MIT"

    assert len(resume.projects) == 1
    assert resume.projects[0].title == "Resume Analyzer"
    assert len(resume.certifications) == 1
    assert (
      resume.certifications[0].name
      == "AWS Certified Cloud Practitioner"
    )


def test_builder_does_not_modify_extracted_lists():

    builder = ResumeBuilder()

    skills = ["Python", "SQL"]

    resume = builder.build(
        skills=skills,
    )

    resume.skills.append("Docker")

    assert skills == ["Python", "SQL"]


def test_builder_strips_profile_strings():

    builder = ResumeBuilder()

    resume = builder.build(
        current_title="  Data Scientist  ",
        domain_category="  Data Science  ",
        summary="  Experienced candidate  ",
    )

    assert resume.current_title == "Data Scientist"
    assert resume.domain_category == "Data Science"
    assert resume.summary == "Experienced candidate"


def test_builder_preserves_metadata():

    builder = ResumeBuilder()

    resume = builder.build(
        metadata={
            "total_blocks_processed": 25,
            "sections_detected": [
                "contact",
                "experience",
            ],
        },
    )

    assert resume.metadata["total_blocks_processed"] == 25
    assert "experience" in resume.metadata["sections_detected"]


def test_builder_adds_source_filename():

    builder = ResumeBuilder()

    resume = builder.build(
        source_file=r"D:\resumes\candidate_resume.pdf",
    )

    assert resume.metadata["source_file"] == "candidate_resume.pdf"


def test_builder_does_not_override_existing_source_filename():

    builder = ResumeBuilder()

    resume = builder.build(
        source_file=r"D:\resumes\candidate_resume.pdf",
        metadata={
            "source_file": "original.pdf",
        },
    )

    assert resume.metadata["source_file"] == "original.pdf"