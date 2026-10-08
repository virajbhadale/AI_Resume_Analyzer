from src.inference.candidates import (
    CandidateStatus,
    ExperienceCandidate,
)

from src.inference.evidence_models import (
    AnchorEvidence,
    AnchorType,
    BoundaryType,
)

from src.inference.segmentation import (
    ExperienceSegment,
    ExperienceSegmenter,
    SegmentStatus,
)


def make_candidate(
    candidate_id,
    block_id,
    role=None,
    company=None,
    start_date=None,
    end_date=None,
):
    return ExperienceCandidate(
        candidate_id=candidate_id,
        block_ids=[block_id],
        role=role,
        company=company,
        start_date=start_date,
        end_date=end_date,
    )


def test_segment_defaults():

    segment = ExperienceSegment(
        segment_id="segment-1"
    )

    assert segment.segment_id == "segment-1"
    assert segment.block_ids == []
    assert segment.anchor_block_ids == []
    assert segment.status == (
        SegmentStatus.PROVISIONAL
    )


def test_add_block_prevents_duplicates():

    segment = ExperienceSegment(
        segment_id="segment-1"
    )

    segment.add_block("block-1")
    segment.add_block("block-1")

    assert segment.block_ids == ["block-1"]


def test_add_anchor_block_prevents_duplicates():

    segment = ExperienceSegment(
        segment_id="segment-1"
    )

    segment.add_anchor_block("block-1")
    segment.add_anchor_block("block-1")

    assert segment.anchor_block_ids == [
        "block-1"
    ]


def test_segment_anchor_properties():

    segment = ExperienceSegment(
        segment_id="segment-1",
        anchor_block_ids=["block-1"],
    )

    assert segment.has_anchor is True
    assert segment.block_count == 0


def test_role_boundary():

    previous = make_candidate(
        "candidate-1",
        "block-1",
        role="Data Analyst Intern",
        company="ComplyHappy",
        start_date="2023-06-19",
        end_date="2023-08-11",
    )

    current = make_candidate(
        "candidate-2",
        "block-2",
        role="AI Engineer Intern",
        company="Marva AI",
        start_date="2025-01-06",
        end_date="2025-05-17",
    )

    segmenter = ExperienceSegmenter()

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.boundary_type == (
        BoundaryType.NEW_ROLE
    )

    assert boundary.score >= 0.80


def test_company_boundary():

    previous = make_candidate(
        "candidate-1",
        "block-1",
        role="Software Engineer",
        company="Company A",
    )

    current = make_candidate(
        "candidate-2",
        "block-2",
        role="Software Engineer",
        company="Company B",
    )

    segmenter = ExperienceSegmenter()

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.boundary_type == (
        BoundaryType.NEW_COMPANY
    )


def test_date_boundary():

    previous = make_candidate(
        "candidate-1",
        "block-1",
        role="Software Engineer",
        start_date="2023-01-01",
        end_date="2023-06-01",
    )

    current = make_candidate(
        "candidate-2",
        "block-2",
        role="Software Engineer",
        start_date="2024-01-01",
        end_date="2024-06-01",
    )

    segmenter = ExperienceSegmenter()

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.boundary_type == (
        BoundaryType.NEW_DATE_RANGE
    )


def test_same_experience_without_strong_boundary():

    previous = make_candidate(
        "candidate-1",
        "block-1",
        role="Software Engineer",
        company="Company A",
    )

    current = make_candidate(
        "candidate-2",
        "block-2",
    )

    segmenter = ExperienceSegmenter()

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.boundary_type == (
        BoundaryType.SAME_EXPERIENCE
    )


def test_group_candidates():

    candidates = [
        make_candidate(
            "candidate-1",
            "block-1",
            role="Data Analyst Intern",
            company="ComplyHappy",
            start_date="2023-06-19",
            end_date="2023-08-11",
        ),
        make_candidate(
            "candidate-2",
            "block-2",
            role="AI Engineer Intern",
            company="Marva AI",
            start_date="2025-01-06",
            end_date="2025-05-17",
        ),
    ]

    segmenter = ExperienceSegmenter()

    groups = segmenter.group_candidates(
        candidates
    )

    assert len(groups) == 2


def test_segment_creates_provisional_segments():

    candidates = [
        make_candidate(
            "candidate-1",
            "block-1",
            role="Data Analyst Intern",
            company="ComplyHappy",
            start_date="2023-06-19",
            end_date="2023-08-11",
        ),
        make_candidate(
            "candidate-2",
            "block-2",
            role="AI Engineer Intern",
            company="Marva AI",
            start_date="2025-01-06",
            end_date="2025-05-17",
        ),
    ]

    segmenter = ExperienceSegmenter()

    result = segmenter.segment(
        candidates
    )

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1


def test_segment_preserves_block_ids():

    candidates = [
        make_candidate(
            "candidate-1",
            "block-1",
            role="Data Analyst",
            company="Company A",
        ),
        make_candidate(
            "candidate-2",
            "block-2",
        ),
    ]

    segmenter = ExperienceSegmenter()

    result = segmenter.segment(
        candidates
    )

    assert "block-1" in result.segments[0].block_ids
    assert "block-1" in (
        result.segments[0].anchor_block_ids
    )


def test_boundary_is_preserved_between_segments():

    candidates = [
        make_candidate(
            "candidate-1",
            "block-1",
            role="Data Analyst",
            company="Company A",
        ),
        make_candidate(
            "candidate-2",
            "block-2",
            role="AI Engineer",
            company="Company B",
        ),
    ]

    segmenter = ExperienceSegmenter()

    result = segmenter.segment(
        candidates
    )

    assert (
        result.segments[0].boundary_after
        is not None
    )

    assert (
        result.segments[1].boundary_before
        is not None
    )


def test_empty_segmentation():

    segmenter = ExperienceSegmenter()

    result = segmenter.segment([])

    assert result.segments == []
    assert result.boundaries == []