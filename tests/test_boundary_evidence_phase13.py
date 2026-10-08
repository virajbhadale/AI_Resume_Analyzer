from src.inference.segmentation import ExperienceSegmenter
from src.inference.evidence_models import BoundaryType
from src.inference.candidates import ExperienceCandidate


def make_candidate(
    candidate_id,
    role=None,
    company=None,
    start_date=None,
    end_date=None,
    is_current=False,
):
    return ExperienceCandidate(
        candidate_id=candidate_id,
        block_ids=[candidate_id],
        role=role,
        company=company,
        location=None,
        start_date=start_date,
        end_date=end_date,
        is_current=is_current,
    )


def test_same_company_different_roles_creates_new_role_boundary():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Senior Data Scientist",
        company="ABC Corp",
        start_date="2024",
        is_current=True,
    )

    current = make_candidate(
        "c2",
        role="Data Scientist",
        company="ABC Corp",
        start_date="2022",
        end_date="2024",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.boundary_type == BoundaryType.NEW_ROLE
    assert boundary.score >= 0.95

    assert boundary.supporting_evidence
    assert boundary.contradictory_evidence

    assert any(
        signal.metadata.get("boundary_signal")
        == "role_change"
        for signal in boundary.supporting_evidence
    )

    assert any(
        signal.metadata.get("boundary_signal")
        == "same_company"
        for signal in boundary.contradictory_evidence
    )


def test_different_role_and_company_preserves_both_signals():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Data Scientist",
        company="Company A",
        start_date="2022",
        end_date="2024",
    )

    current = make_candidate(
        "c2",
        role="ML Engineer",
        company="Company B",
        start_date="2024",
        is_current=True,
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.boundary_type == BoundaryType.NEW_ROLE
    assert boundary.score >= 0.99

    supporting_types = {
        signal.metadata.get("boundary_signal")
        for signal in boundary.supporting_evidence
    }

    assert "role_change" in supporting_types
    assert "company_change" in supporting_types
    assert "date_range_change" in supporting_types


def test_same_role_and_company_supports_continuation():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Data Scientist",
        company="ABC Corp",
    )

    current = make_candidate(
        "c2",
        role="Data Scientist",
        company="ABC Corp",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert (
        boundary.boundary_type
        == BoundaryType.DESCRIPTION_CONTINUATION
    )

    assert boundary.score < 0.80

    assert any(
        signal.metadata.get("boundary_signal")
        == "same_company"
        for signal in boundary.contradictory_evidence
    )


def test_distinct_dates_are_preserved_as_boundary_evidence():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Data Scientist",
        company="ABC Corp",
        start_date="2020",
        end_date="2022",
    )

    current = make_candidate(
        "c2",
        role="Data Scientist",
        company="ABC Corp",
        start_date="2022",
        end_date="2024",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    # Date evidence exists, but the exact classification remains
    # subordinate to role/company reasoning.
    assert any(
        signal.metadata.get("boundary_signal")
        == "date_range_change"
        for signal in boundary.supporting_evidence
    )


def test_unknown_metadata_does_not_create_false_boundary():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Data Scientist",
    )

    current = make_candidate(
        "c2",
        role="Data Scientist",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.score < 0.80