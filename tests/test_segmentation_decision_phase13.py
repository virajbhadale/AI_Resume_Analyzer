from src.inference.segmentation import ExperienceSegmenter
from src.inference.evidence_models import BoundaryType
from src.inference.candidates import ExperienceCandidate


def make_candidate(
    candidate_id,
    role=None,
    company=None,
    location=None,
    start_date=None,
    end_date=None,
    is_current=None,
):
    return ExperienceCandidate(
        candidate_id=candidate_id,
        block_ids=[candidate_id],
        role=role,
        company=company,
        location=location,
        start_date=start_date,
        end_date=end_date,
        is_current=is_current,
    )


# ============================================================================
# Strong boundaries
# ============================================================================


def test_clear_role_company_change_creates_two_segments():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Data Scientist",
            company="Company A",
            start_date="2022",
            end_date="2024",
        ),
        make_candidate(
            "c2",
            role="ML Engineer",
            company="Company B",
            start_date="2024",
            is_current=True,
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1

    assert (
        result.boundaries[0].boundary_type
        == BoundaryType.NEW_ROLE
    )


def test_same_company_different_roles_still_creates_two_segments():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Senior Data Scientist",
            company="ABC Technologies",
            start_date="2024",
            is_current=True,
        ),
        make_candidate(
            "c2",
            role="Data Scientist",
            company="ABC Technologies",
            start_date="2022",
            end_date="2024",
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1

    boundary = result.boundaries[0]

    assert boundary.boundary_type == BoundaryType.NEW_ROLE
    assert boundary.score >= 0.80


def test_same_role_different_company_creates_two_segments():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Data Scientist",
            company="Company A",
            start_date="2022",
            end_date="2024",
        ),
        make_candidate(
            "c2",
            role="Data Scientist",
            company="Company B",
            start_date="2024",
            is_current=True,
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1

    assert (
        result.boundaries[0].boundary_type
        == BoundaryType.NEW_COMPANY
    )


# ============================================================================
# Continuation
# ============================================================================


def test_same_role_company_without_boundary_keeps_one_segment():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Data Scientist",
            company="ABC Technologies",
        ),
        make_candidate(
            "c2",
            role="Data Scientist",
            company="ABC Technologies",
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 1
    assert len(result.boundaries) == 0


def test_location_only_change_does_not_split_segment():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Data Scientist",
            location="Pune, India",
        ),
        make_candidate(
            "c2",
            role="Data Scientist",
            location="Mumbai, India",
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 1
    assert len(result.boundaries) == 0


# ============================================================================
# Date boundary
# ============================================================================


def test_distinct_date_ranges_can_split_candidates():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Data Scientist",
            company="ABC Technologies",
            start_date="2020",
            end_date="2022",
        ),
        make_candidate(
            "c2",
            role="Data Scientist",
            company="ABC Technologies",
            start_date="2023",
            end_date="2024",
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1

    boundary = result.boundaries[0]

    assert (
        boundary.boundary_type
        == BoundaryType.NEW_DATE_RANGE
    )
    assert boundary.score >= 0.80


# ============================================================================
# Missing metadata
# ============================================================================


def test_missing_metadata_does_not_create_false_split():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
        ),
        make_candidate(
            "c2",
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 1
    assert len(result.boundaries) == 0


def test_missing_company_does_not_create_false_company_split():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Data Analyst Intern",
            company=None,
            start_date="2023",
            end_date="2023",
        ),
        make_candidate(
            "c2",
            role="Data Analyst",
            company=None,
            start_date="2022",
            end_date="2023",
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 2

    assert (
        result.boundaries[0].boundary_type
        == BoundaryType.NEW_ROLE
    )


# ============================================================================
# Multiple sequential experiences
# ============================================================================


def test_three_distinct_experiences_create_three_segments():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Data Analyst",
            company="Company A",
            start_date="2020",
            end_date="2021",
        ),
        make_candidate(
            "c2",
            role="Data Scientist",
            company="Company B",
            start_date="2021",
            end_date="2023",
        ),
        make_candidate(
            "c3",
            role="ML Engineer",
            company="Company C",
            start_date="2023",
            is_current=True,
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 3
    assert len(result.boundaries) == 2


# ============================================================================
# Segment provenance
# ============================================================================


def test_split_segments_preserve_candidate_and_block_provenance():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "candidate-1",
            role="Data Scientist",
            company="Company A",
            start_date="2022",
            end_date="2024",
        ),
        make_candidate(
            "candidate-2",
            role="ML Engineer",
            company="Company B",
            start_date="2024",
            is_current=True,
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 2

    assert (
        result.segments[0].candidate_id
        == "candidate-1"
    )

    assert (
        result.segments[1].candidate_id
        == "candidate-2"
    )

    assert (
        result.segments[0].block_ids
        == ["candidate-1"]
    )

    assert (
        result.segments[1].block_ids
        == ["candidate-2"]
    )


# ============================================================================
# Boundary provenance
# ============================================================================


def test_boundary_provenance_is_attached_to_both_adjacent_segments():
    segmenter = ExperienceSegmenter()

    candidates = [
        make_candidate(
            "c1",
            role="Data Scientist",
            company="Company A",
        ),
        make_candidate(
            "c2",
            role="ML Engineer",
            company="Company B",
        ),
    ]

    result = segmenter.segment(candidates)

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1

    boundary = result.boundaries[0]

    assert (
        result.segments[1].boundary_before
        is not None
    )

    assert (
        result.segments[0].boundary_after
        is not None
    )

    assert (
        result.segments[1].boundary_before.boundary_type
        == boundary.boundary_type
    )

    assert (
        result.segments[0].boundary_after.boundary_type
        == boundary.boundary_type
    )