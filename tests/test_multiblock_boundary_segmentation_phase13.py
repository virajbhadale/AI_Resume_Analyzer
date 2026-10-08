"""
Phase 13D — Multi-block boundary → segmentation integration.

Purpose:
    Verify that multi-block experience candidates can flow through:

        BlockEvidence
            ↓
        CandidateBuilder
            ↓
        ExperienceCandidate[]
            ↓
        BoundaryEvidence
            ↓
        ExperienceSegment[]

without metadata leaking across experience boundaries.

This is intentionally an integration-level test layer.
It does not modify production segmentation logic.
"""

from src.inference.candidate_builder import MultiBlockCandidateBuilder
from src.inference.segmentation import ExperienceSegmenter

from tests.test_multiblock_candidate_builder import block


def _build_candidates(blocks):
    """
    Build evidence-based experience candidates using the
    repository's existing MultiBlockCandidateBuilder.
    """
    return MultiBlockCandidateBuilder().build_from_blocks(blocks)


def _segment(candidates):
    """
    Run the existing segmentation implementation.
    """
    return ExperienceSegmenter().segment(candidates)


def test_role_company_date_split_across_blocks_is_one_segment():
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

    candidates = _build_candidates(blocks)
    result = _segment(candidates)

    assert len(result.segments) == 1

    segment = result.segments[0]

    assert segment.role == "Machine Learning Engineer"
    assert segment.company == "OpenAI"
    assert segment.start_date == "Jan 2024"
    assert segment.is_current is True

    assert set(segment.block_ids) == {"b1", "b2", "b3", "b4"}


def test_two_sequential_multiblock_jobs_remain_two_segments():
    blocks = [
        # Experience 1
        block("b1", "Data Analyst", 1, bold=True, y=100),
        block("b2", "Acme Analytics", 2, y=125),
        block("b3", "2022 - 2023", 3, y=150),
        block("b4", "Built reporting dashboards.", 4, y=175),

        # Experience 2
        block("b5", "AI Engineer", 5, bold=True, y=240),
        block("b6", "Beta AI", 6, y=265),
        block("b7", "2024 - Present", 7, y=290),
        block("b8", "Built ML services.", 8, y=315),
    ]

    candidates = _build_candidates(blocks)
    result = _segment(candidates)

    assert len(result.segments) == 2

    first, second = result.segments

    assert first.role == "Data Analyst"
    assert first.company == "Acme Analytics"

    assert second.role == "AI Engineer"
    assert second.company == "Beta AI"

    assert set(first.block_ids).isdisjoint(second.block_ids)


def test_same_company_different_roles_remain_two_segments():
    blocks = [
        block("b1", "Software Engineer", 1, bold=True, y=100),
        block("b2", "Acme Corp", 2, y=125),
        block("b3", "2021 - 2022", 3, y=150),

        block("b4", "Senior Software Engineer", 4, bold=True, y=220),
        block("b5", "Acme Corp", 5, y=245),
        block("b6", "2022 - Present", 6, y=270),
    ]

    candidates = _build_candidates(blocks)
    result = _segment(candidates)

    assert len(result.segments) == 2

    assert result.segments[0].role == "Software Engineer"
    assert result.segments[0].company == "Acme Corp"

    assert result.segments[1].role == "Senior Software Engineer"
    assert result.segments[1].company == "Acme Corp"


def test_same_role_different_companies_remain_two_segments():
    blocks = [
        block("b1", "Data Scientist", 1, bold=True, y=100),
        block("b2", "Alpha Labs", 2, y=125),
        block("b3", "2021 - 2022", 3, y=150),

        block("b4", "Data Scientist", 4, bold=True, y=220),
        block("b5", "Beta Analytics", 5, y=245),
        block("b6", "2023 - Present", 6, y=270),
    ]

    candidates = _build_candidates(blocks)
    result = _segment(candidates)

    assert len(result.segments) == 2

    assert result.segments[0].company == "Alpha Labs"
    assert result.segments[1].company == "Beta Analytics"


def test_description_blocks_remain_with_preceding_experience():
    blocks = [
        block("b1", "ML Engineer", 1, bold=True, y=100),
        block("b2", "OpenAI", 2, y=125),
        block("b3", "2024 - Present", 3, y=150),
        block("b4", "Developed inference pipelines.", 4, y=175),
        block("b5", "Improved model serving latency.", 5, y=200),
    ]

    candidates = _build_candidates(blocks)
    result = _segment(candidates)

    assert len(result.segments) == 1

    segment = result.segments[0]

    assert set(segment.block_ids) == {
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
    }


def test_missing_company_does_not_create_spurious_boundary():
    blocks = [
        block("b1", "Research Assistant", 1, bold=True, y=100),
        block("b2", "2023 - Present", 2, y=125),
        block("b3", "Conducted ML research.", 3, y=150),
    ]

    candidates = _build_candidates(blocks)
    result = _segment(candidates)

    assert len(result.segments) == 1

    segment = result.segments[0]

    assert segment.role == "Research Assistant"
    assert segment.company is None


def test_location_change_alone_does_not_create_new_segment():
    blocks = [
        block("b1", "Data Analyst", 1, bold=True, y=100),
        block("b2", "Acme Corp", 2, y=125),
        block("b3", "Pune, India", 3, y=150),
        block("b4", "2023 - Present", 4, y=175),
        block("b5", "Built analytics dashboards.", 5, y=200),
    ]

    candidates = _build_candidates(blocks)
    result = _segment(candidates)

    assert len(result.segments) == 1


def test_multi_page_continuation_remains_one_experience():
    blocks = [
        block(
            "b1",
            "Machine Learning Engineer",
            1,
            bold=True,
            page=1,
            y=100,
        ),
        block(
            "b2",
            "OpenAI",
            2,
            page=1,
            y=125,
        ),
        block(
            "b3",
            "2024 - Present",
            3,
            page=1,
            y=150,
        ),
        block(
            "b4",
            "Built production ML infrastructure.",
            4,
            page=1,
            y=175,
        ),
        block(
            "b5",
            "Continued optimizing inference systems.",
            5,
            page=2,
            y=100,
        ),
    ]

    candidates = _build_candidates(blocks)
    result = _segment(candidates)

    assert len(result.segments) == 1

    assert set(result.segments[0].block_ids) == {
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
    }


def test_section_transition_does_not_consume_next_section():
    blocks = [
        block(
            "b1",
            "Software Engineer",
            1,
            bold=True,
            y=100,
            section="experience",
        ),
        block(
            "b2",
            "Acme Corp",
            2,
            y=125,
            section="experience",
        ),
        block(
            "b3",
            "2024 - Present",
            3,
            y=150,
            section="experience",
        ),
        block(
            "b4",
            "Projects",
            4,
            y=200,
            section="projects",
        ),
        block(
            "b5",
            "Built an AI project.",
            5,
            y=225,
            section="projects",
        ),
    ]

    candidates = _build_candidates(blocks)

    # Candidate construction must not pull project blocks
    # into the experience candidate.
    experience_block_ids = {
        block_id
        for candidate in candidates
        for block_id in candidate.block_ids
    }

    assert "b4" not in experience_block_ids
    assert "b5" not in experience_block_ids

    result = _segment(candidates)

    assert len(result.segments) == 1