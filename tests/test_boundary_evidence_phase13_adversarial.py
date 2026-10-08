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
    block_ids=None,
):
    return ExperienceCandidate(
        candidate_id=candidate_id,
        block_ids=block_ids or [candidate_id],
        role=role,
        company=company,
        location=location,
        start_date=start_date,
        end_date=end_date,
        is_current=is_current,
    )


def boundary_signals(boundary):
    return {
        signal.metadata.get("boundary_signal")
        for signal in (
            boundary.supporting_evidence
            + boundary.contradictory_evidence
        )
    }


# ============================================================================
# Role fragmentation / continuation
# ============================================================================


def test_role_fragment_candidates_do_not_create_false_strong_boundary():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Senior Machine Learning",
        company=None,
    )

    current = make_candidate(
        "c2",
        role="Engineer",
        company="ABC Technologies",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    # These candidates are ambiguous at this abstraction level.
    # The detector must not invent company/date evidence.
    assert "company_change" not in boundary_signals(boundary)
    assert "date_range_change" not in boundary_signals(boundary)


# ============================================================================
# Same company / different role
# ============================================================================


def test_same_company_different_role_remains_new_role_boundary():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Senior Data Scientist",
        company="ABC Technologies",
        start_date="2024",
        is_current=True,
    )

    current = make_candidate(
        "c2",
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2022",
        end_date="2024",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.boundary_type == BoundaryType.NEW_ROLE
    assert boundary.score >= 0.95

    signals = boundary_signals(boundary)

    assert "role_change" in signals
    assert "same_company" in signals


# ============================================================================
# Same role / different company
# ============================================================================


def test_same_role_different_company_creates_company_boundary():
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
        role="Data Scientist",
        company="Company B",
        start_date="2024",
        is_current=True,
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.boundary_type == BoundaryType.NEW_COMPANY
    assert boundary.score >= 0.90

    signals = boundary_signals(boundary)

    assert "company_change" in signals
    assert "same_role" in signals


# ============================================================================
# Missing company
# ============================================================================


def test_missing_company_does_not_create_company_change():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Data Analyst Intern",
        company=None,
        location="Pune, India",
        start_date="2023",
        end_date="2023",
    )

    current = make_candidate(
        "c2",
        role="Data Analyst",
        company=None,
        location="Pune, India",
        start_date="2022",
        end_date="2023",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert "company_change" not in boundary_signals(boundary)
    assert boundary.boundary_type == BoundaryType.NEW_ROLE


# ============================================================================
# Missing role
# ============================================================================


def test_missing_role_does_not_create_role_change():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role=None,
        company="Company A",
        start_date="2022",
        end_date="2024",
    )

    current = make_candidate(
        "c2",
        role=None,
        company="Company B",
        start_date="2024",
        is_current=True,
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert "role_change" not in boundary_signals(boundary)
    assert "company_change" in boundary_signals(boundary)
    assert boundary.boundary_type == BoundaryType.NEW_COMPANY


# ============================================================================
# Current experience
# ============================================================================


def test_current_role_is_compared_without_treating_current_as_new_role():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Data Scientist",
        company="ABC",
        start_date="2022",
        end_date="2024",
    )

    current = make_candidate(
        "c2",
        role="Data Scientist",
        company="ABC",
        start_date="2024",
        is_current=True,
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert "role_change" not in boundary_signals(boundary)
    assert "same_role" in boundary_signals(boundary)
    assert "same_company" in boundary_signals(boundary)
    assert "date_range_change" in boundary_signals(boundary)


# ============================================================================
# Same metadata / continuation
# ============================================================================


def test_same_role_company_without_dates_is_continuation():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Machine Learning Engineer",
        company="ABC Technologies",
    )

    current = make_candidate(
        "c2",
        role="Machine Learning Engineer",
        company="ABC Technologies",
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

    signals = boundary_signals(boundary)

    assert "same_role" in signals
    assert "same_company" in signals
    assert "experience_continuation" in signals


# ============================================================================
# Date-only distinction
# ============================================================================


def test_distinct_dates_are_preserved_even_when_role_and_company_match():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2020",
        end_date="2022",
    )

    current = make_candidate(
        "c2",
        role="Data Scientist",
        company="ABC Technologies",
        start_date="2022",
        end_date="2024",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    signals = boundary_signals(boundary)

    assert "same_role" in signals
    assert "same_company" in signals
    assert "date_range_change" in signals


# ============================================================================
# No metadata
# ============================================================================


def test_no_metadata_does_not_produce_false_positive_boundary():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
    )

    current = make_candidate(
        "c2",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert boundary.score < 0.80
    assert boundary.boundary_type in {
        BoundaryType.SAME_EXPERIENCE,
        BoundaryType.DESCRIPTION_CONTINUATION,
    }


# ============================================================================
# Location alone
# ============================================================================


def test_location_change_alone_does_not_create_new_experience():
    segmenter = ExperienceSegmenter()

    previous = make_candidate(
        "c1",
        role="Data Scientist",
        company=None,
        location="Pune, India",
    )

    current = make_candidate(
        "c2",
        role="Data Scientist",
        company=None,
        location="Mumbai, India",
    )

    boundary = segmenter.detect_boundary(
        previous,
        current,
    )

    assert "company_change" not in boundary_signals(boundary)
    assert "role_change" not in boundary_signals(boundary)

    assert boundary.score < 0.80