"""Tests for the public ExperienceExtractor."""

from src.document_models import DocumentBlock
from src.extraction.experience import ExperienceExtractor


def test_extract_single_experience():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Senior Data Scientist\n"
                "Nexus Analytics\n"
                "Jan 2023 – Present\n"
                "Built machine learning models using Python and Scikit-Learn."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    results = extractor.extract(blocks)

    assert len(results) == 1

    experience = results[0]

    assert experience.role == "Senior Data Scientist"
    assert experience.company == "Nexus Analytics"
    assert experience.start_date == "Jan 2023"
    assert experience.end_date == "Present"
    assert experience.is_current is True


def test_extract_multiple_experiences():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Senior Data Scientist\n"
                "Nexus Analytics\n"
                "Jan 2023 – Present\n"
                "Built predictive models using Python and Scikit-Learn."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b2",
            text=(
                "Machine Learning Engineer\n"
                "Apex AI Solutions\n"
                "Jun 2021 – Dec 2022\n"
                "Built inference services using FastAPI and Docker."
            ),
            is_bold=True,
            section="experience",
        ),
    ]

    results = extractor.extract(blocks)

    assert len(results) == 2

    assert results[0].role == "Senior Data Scientist"
    assert results[0].company == "Nexus Analytics"

    assert results[1].role == "Machine Learning Engineer"
    assert results[1].company == "Apex AI Solutions"

def test_extractor_preserves_same_company_different_roles():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Software Engineer\n"
                "Google\n"
                "2023 - 2024\n"
                "Built backend services."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b2",
            text=(
                "Senior Software Engineer\n"
                "Google\n"
                "2024 - Present\n"
                "Led distributed systems development."
            ),
            is_bold=True,
            section="experience",
        ),
    ]

    results = extractor.extract(blocks)

    assert len(results) == 2

    assert results[0].role == "Software Engineer"
    assert results[0].company == "Google"
    assert results[0].start_date == "2023"
    assert results[0].end_date == "2024"

    assert results[1].role == "Senior Software Engineer"
    assert results[1].company == "Google"
    assert results[1].start_date == "2024"
    assert results[1].end_date == "Present"

def test_extractor_preserves_same_role_company_different_periods():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Software Engineer\n"
                "ABC Technologies\n"
                "Jan 2022 - Jun 2022\n"
                "Built backend APIs and data services."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b2",
            text=(
                "Software Engineer\n"
                "ABC Technologies\n"
                "Jan 2024 - Jun 2024\n"
                "Designed machine learning infrastructure."
            ),
            is_bold=True,
            section="experience",
        ),
    ]

    results = extractor.extract(blocks)

    assert len(results) == 2

    assert results[0].role == "Software Engineer"
    assert results[0].company == "ABC Technologies"
    assert results[0].start_date == "Jan 2022"
    assert results[0].end_date == "Jun 2022"
    assert (
        "Built backend APIs and data services."
        in results[0].bullet_points
    )

    assert results[1].role == "Software Engineer"
    assert results[1].company == "ABC Technologies"
    assert results[1].start_date == "Jan 2024"
    assert results[1].end_date == "Jun 2024"
    assert (
        "Designed machine learning infrastructure."
        in results[1].bullet_points
    )

def test_extractor_preserves_dated_then_datedless_experience():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Data Analyst\n"
                "ABC Corp\n"
                "Jan 2023 - Jun 2023\n"
                "Built automated reporting dashboards."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b2",
            text=(
                "Machine Learning Intern\n"
                "XYZ Labs"
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b3",
            text=(
                "Developed machine learning models and "
                "data preprocessing pipelines."
            ),
            section="experience",
        ),
    ]

    results = extractor.extract(blocks)

    assert len(results) == 2

    assert results[0].role == "Data Analyst"
    assert results[0].company == "ABC Corp"
    assert results[0].start_date == "Jan 2023"
    assert results[0].end_date == "Jun 2023"

    assert results[1].role == "Machine Learning Intern"
    assert results[1].company == "XYZ Labs"
    assert results[1].start_date is None
    assert results[1].end_date is None

    assert (
        "Developed machine learning models and "
        "data preprocessing pipelines."
        in results[1].bullet_points
    )        

def test_experience_extracts_bullet_points():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Machine Learning Engineer\n"
                "Apex AI Solutions\n"
                "Jun 2021 – Dec 2022\n"
                "• Built FastAPI inference service.\n"
                "• Trained XGBoost models for credit risk.\n"
                "• Automated model evaluation pipelines."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    results = extractor.extract(blocks)

    assert len(results) == 1

    experience = results[0]

    assert len(experience.bullet_points) >= 3
    assert any("FastAPI" in bullet for bullet in experience.bullet_points)
    assert any("XGBoost" in bullet for bullet in experience.bullet_points)


def test_experience_extracts_technologies():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Machine Learning Engineer\n"
                "Apex AI Solutions\n"
                "2021 – 2022\n"
                "Built FastAPI services using Python and Docker."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    results = extractor.extract(blocks)

    assert len(results) == 1

    technologies = results[0].technologies_mentioned

    assert "Python" in technologies
    assert "FastAPI" in technologies
    assert "Docker" in technologies


def test_experience_confidence_is_calculated():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Data Scientist\n"
                "Nexus Analytics\n"
                "Jan 2023 – Present\n"
                "Built predictive machine learning models.\n"
                "Improved model performance by 20%."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    results = extractor.extract(blocks)

    assert len(results) == 1

    confidence = results[0].confidence

    assert 0.0 <= confidence <= 0.99
    assert confidence > 0.50


def test_empty_experience_returns_empty_list():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text="ALEXANDER CHEN",
            section="contact",
        ),
        DocumentBlock(
            id="b2",
            text="Python, SQL, Pandas",
            section="skills",
        ),
    ]

    results = extractor.extract(blocks)

    assert results == []


def test_non_experience_sections_are_ignored():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Machine Learning Engineer\n"
                "Apex AI Solutions\n"
                "2021 – 2022\n"
                "Built production systems."
            ),
            section="projects",
        )
    ]

    results = extractor.extract(blocks)

    assert results == []


def test_experience_fallback_detection():
    """
    ExperienceExtractor should attempt evidence-based fallback
    when the canonical experience section is missing.
    """

    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Machine Learning Engineer\n"
                "Apex AI Solutions\n"
                "2021 – 2022\n"
                "Built production inference systems."
            ),
            is_bold=True,
            section=None,
        )
    ]

    results = extractor.extract(blocks)

    assert len(results) >= 1
    assert "Machine Learning Engineer" in results[0].role

def test_extractor_metadata_normalization():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="metadata_1",
            text=(
                "  Senior Data Scientist  \n"
                "  Nexus Analytics  \n"
                "Jan 2023 - Present\n"
                "Built predictive models."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    results = extractor.extract(blocks)

    assert len(results) == 1

    experience = results[0]

    assert experience.role == "Senior Data Scientist"
    assert experience.company == "Nexus Analytics"
    assert experience.start_date == "Jan 2023"
    assert experience.end_date == "Present"
    assert experience.is_current is True

    assert "metadata_resolution" in experience.evidence

def test_extractor_normalizes_current_status():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="current_1",
            text=(
                "AI Engineer\n"
                "Marva AI\n"
                "2025 - Current\n"
                "Developed AI systems."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    results = extractor.extract(blocks)

    assert len(results) == 1

    experience = results[0]

    assert experience.end_date == "Present"
    assert experience.is_current is True

def test_metadata_resolution_preserves_separate_experiences():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="metadata_repeat_1",
            text=(
                "Software Engineer\n"
                "ABC Technologies\n"
                "Jan 2022 - Jun 2022\n"
                "Built backend APIs."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="metadata_repeat_2",
            text=(
                "Software Engineer\n"
                "ABC Technologies\n"
                "Jan 2024 - Jun 2024\n"
                "Designed ML infrastructure."
            ),
            is_bold=True,
            section="experience",
        ),
    ]

    results = extractor.extract(blocks)

    assert len(results) == 2

    first = results[0]
    second = results[1]

    assert first.role == second.role
    assert first.company == second.company

    assert first.start_date == "Jan 2022"
    assert first.end_date == "Jun 2022"

    assert second.start_date == "Jan 2024"
    assert second.end_date == "Jun 2024"

    assert first.id != second.id

    assert "Built backend APIs." in first.bullet_points
    assert "Designed ML infrastructure." in second.bullet_points

def test_metadata_resolution_preserves_missing_metadata():
    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="metadata_missing",
            text=(
                "Machine Learning Intern\n"
                "XYZ Labs"
            ),
            is_bold=True,
            section="experience",
        )
    ]

    results = extractor.extract(blocks)

    assert len(results) == 1

    experience = results[0]

    assert experience.role == "Machine Learning Intern"
    assert experience.company == "XYZ Labs"

    assert experience.location is None
    assert experience.start_date is None
    assert experience.end_date is None
    assert experience.is_current is False

    assert "metadata_resolution" in experience.evidence                