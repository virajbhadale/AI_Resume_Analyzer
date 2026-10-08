from src.extraction.experience import ExperienceExtractor
from src.document_models import (
    BBox,
    DocumentBlock,
)


def block(
    block_id: str,
    text: str,
    reading_order: int,
    *,
    bold: bool = False,
):
    return DocumentBlock(
        id=block_id,
        page=1,
        page_width=600.0,
        page_height=800.0,
        text=text,
        bbox=BBox(
            x0=40.0,
            y0=100.0 + reading_order * 25.0,
            x1=500.0,
            y1=120.0 + reading_order * 25.0,
        ),
        column_id=0,
        reading_order=reading_order,
        is_bold=bold,
        is_heading=False,
        block_type="text",
        source_type="test",
        extraction_method="test",
        section="experience",
        confidence=1.0,
        metadata={},
    )


def test_evidence_based_path_materializes_episode():
    blocks = [
        block(
            "b1",
            "Machine Learning Engineer",
            1,
            bold=True,
        ),
        block(
            "b2",
            "OpenAI",
            2,
        ),
        block(
            "b3",
            "Jan 2024 - Present",
            3,
        ),
        block(
            "b4",
            "Built production inference systems.",
            4,
        ),
    ]

    extractor = ExperienceExtractor()

    episodes = extractor.extract_evidence_based(
        blocks
    )

    assert len(episodes) == 1

    episode = episodes[0]

    assert (
        episode.role
        == "Machine Learning Engineer"
    )

    assert episode.company == "OpenAI"

    assert episode.start_date == "Jan 2024"

    assert episode.is_current is True


def test_evidence_based_episode_contains_provenance():
    blocks = [
        block(
            "b1",
            "Data Scientist",
            1,
            bold=True,
        ),
        block(
            "b2",
            "Acme Analytics",
            2,
        ),
        block(
            "b3",
            "2022 - 2024",
            3,
        ),
    ]

    extractor = ExperienceExtractor()

    episodes = extractor.extract_evidence_based(
        blocks
    )

    assert len(episodes) == 1

    evidence = episodes[0].evidence

    assert evidence["provenance_version"] == 1

    assert evidence["source_block_ids"]

    assert evidence["candidate_id"]

    assert "field_evidence" in evidence

    assert "validation" in evidence


def test_evidence_based_path_does_not_cross_non_experience_section():
    blocks = [
        block(
            "b1",
            "Software Engineer",
            1,
            bold=True,
        ),
        block(
            "b2",
            "Acme Corp",
            2,
        ),
        block(
            "b3",
            "2024 - Present",
            3,
        ),
        DocumentBlock(
            id="p1",
            page=1,
            page_width=600.0,
            page_height=800.0,
            text="Resume Analyzer",
            bbox=BBox(
                x0=40.0,
                y0=300.0,
                x1=500.0,
                y1=320.0,
            ),
            column_id=0,
            reading_order=4,
            is_bold=True,
            is_heading=False,
            block_type="text",
            source_type="test",
            extraction_method="test",
            section="projects",
            confidence=1.0,
            metadata={},
        ),
    ]

    extractor = ExperienceExtractor()

    episodes = extractor.extract_evidence_based(
        blocks
    )

    assert len(episodes) == 1

    assert (
        episodes[0].role
        == "Software Engineer"
    )

    assert (
        "Resume Analyzer"
        not in episodes[0].description
    )


def test_existing_extract_path_remains_available():
    extractor = ExperienceExtractor()

    assert callable(
        extractor.extract
    )

    assert callable(
        extractor.extract_evidence_based
    )