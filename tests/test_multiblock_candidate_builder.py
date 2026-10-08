from src.document_models import BBox, DocumentBlock
from src.inference.candidate_builder import MultiBlockCandidateBuilder


def block(
    block_id,
    text,
    reading_order,
    *,
    section="experience",
    page=1,
    column_id=0,
    bold=False,
    y=0,
):
    """
    Helper for creating deterministic DocumentBlock fixtures.
    """

    return DocumentBlock(
        id=block_id,
        text=text,
        page=page,
        page_width=600,
        page_height=800,
        bbox=BBox(
            x0=40,
            y0=y,
            x1=560,
            y1=y + 18,
        ),
        reading_order=reading_order,
        column_id=column_id,
        section=section,
        is_bold=bold,
    )


def test_assembles_role_company_date_description_across_blocks():
    """
    A standard split experience:

        ROLE
        COMPANY
        DATE
        DESCRIPTION

    should become one candidate.
    """

    blocks = [
        block(
            "b1",
            "Machine Learning Engineer",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "OpenAI",
            2,
            y=125,
        ),
        block(
            "b3",
            "Jan 2024 - Present",
            3,
            y=150,
        ),
        block(
            "b4",
            "Built production inference systems.",
            4,
            y=175,
        ),
    ]

    candidates = (
        MultiBlockCandidateBuilder()
        .build_from_blocks(blocks)
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.role == "Machine Learning Engineer"
    assert candidate.company == "OpenAI"
    assert candidate.start_date == "Jan 2024"
    assert candidate.is_current is True

    assert candidate.block_ids == [
        "b1",
        "b2",
        "b3",
        "b4",
    ]

    assert candidate.description_block_ids == [
        "b4",
    ]

    assert (
        candidate.evidence.metadata["assembly"]
        == "multi_block"
    )

    assert (
        candidate.evidence.metadata[
            "field_sources"
        ]["company"]
        == "b2"
    )

    assert (
        candidate.evidence.metadata[
            "field_sources"
        ]["date"]
        == "b3"
    )


def test_does_not_cross_next_role_anchor():
    """
    Two consecutive jobs must remain two candidates.

    Metadata belonging to the second role must not be attached
    to the first role.
    """

    blocks = [
        block(
            "b1",
            "Data Analyst",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Acme Analytics",
            2,
            y=125,
        ),
        block(
            "b3",
            "2023 - 2024",
            3,
            y=150,
        ),
        block(
            "b4",
            "Prepared dashboards.",
            4,
            y=175,
        ),
        block(
            "b5",
            "AI Engineer",
            5,
            bold=True,
            y=220,
        ),
        block(
            "b6",
            "Beta AI",
            6,
            y=245,
        ),
        block(
            "b7",
            "2024 - Present",
            7,
            y=270,
        ),
    ]

    candidates = (
        MultiBlockCandidateBuilder()
        .build_from_blocks(blocks)
    )

    assert len(candidates) == 2

    assert candidates[0].role == "Data Analyst"
    assert candidates[0].company == "Acme Analytics"

    assert "b5" not in candidates[0].block_ids
    assert "b6" not in candidates[0].block_ids
    assert "b7" not in candidates[0].block_ids

    assert candidates[1].role == "AI Engineer"
    assert candidates[1].company == "Beta AI"
    assert candidates[1].is_current is True


def test_can_associate_metadata_before_role_in_local_layout():
    """
    Some layouts visually place company/date before the role in
    extracted reading order.

        COMPANY
        DATE
        ROLE
        DESCRIPTION

    The local evidence should still form one candidate.
    """

    blocks = [
        block(
            "b1",
            "OpenAI",
            1,
            y=100,
        ),
        block(
            "b2",
            "Jan 2024 - Present",
            2,
            y=125,
        ),
        block(
            "b3",
            "Machine Learning Engineer",
            3,
            bold=True,
            y=150,
        ),
        block(
            "b4",
            "Built inference systems.",
            4,
            y=175,
        ),
    ]

    candidates = (
        MultiBlockCandidateBuilder()
        .build_from_blocks(blocks)
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.role == "Machine Learning Engineer"
    assert candidate.company == "OpenAI"
    assert candidate.start_date == "Jan 2024"
    assert candidate.is_current is True

    assert candidate.block_ids == [
        "b1",
        "b2",
        "b3",
        "b4",
    ]


def test_section_boundary_is_not_crossed():
    """
    Experience assembly must not consume blocks explicitly assigned
    to another section.
    """

    blocks = [
        block(
            "b1",
            "Software Engineer",
            1,
            y=100,
        ),
        block(
            "b2",
            "Acme Corp",
            2,
            y=125,
        ),
        block(
            "b3",
            "2024 - Present",
            3,
            y=150,
        ),
        block(
            "b4",
            "Projects",
            4,
            section="projects",
            y=180,
        ),
        block(
            "b5",
            "Built a project",
            5,
            section="projects",
            y=205,
        ),
    ]

    candidates = (
        MultiBlockCandidateBuilder()
        .build_from_blocks(blocks)
    )

    assert len(candidates) == 1

    assert candidates[0].block_ids == [
        "b1",
        "b2",
        "b3",
    ]


def test_action_sentence_is_not_promoted_to_role_or_company():
    """
    Description/action sentences must remain description evidence
    rather than becoming role/company anchors.
    """

    blocks = [
        block(
            "b1",
            "Marketing Manager",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Acme Corp",
            2,
            y=125,
        ),
        block(
            "b3",
            "2023 - Present",
            3,
            y=150,
        ),
        block(
            "b4",
            "Organizes meetings for the manager",
            4,
            y=175,
        ),
        block(
            "b5",
            "Diagnosed and resolved hardware/software issues",
            5,
            y=200,
        ),
    ]

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence(blocks)
    )

    by_id = {
        item.block_id: item
        for item in evidence
    }

    assert by_id["b4"].role is None
    assert by_id["b4"].company is None

    assert by_id["b5"].role is None
    assert by_id["b5"].company is None


def test_block_evidence_preserves_provenance_and_layout():
    """
    BlockEvidence must preserve enough source/layout information for
    downstream explainability and candidate association.
    """

    blocks = [
        block(
            "role",
            "AI Engineer",
            10,
            bold=True,
            page=2,
            column_id=1,
            y=300,
        ),
    ]

    evidence = (
        MultiBlockCandidateBuilder()
        .build_block_evidence(blocks)
    )[0]

    assert evidence.block_id == "role"
    assert evidence.page == 2
    assert evidence.column_id == 1
    assert evidence.reading_order == 10
    assert evidence.is_bold is True

    assert evidence.bbox is not None
    assert evidence.bbox["y0"] == 300

    assert (
        evidence.metadata["is_anchor_head"]
        is False
    )