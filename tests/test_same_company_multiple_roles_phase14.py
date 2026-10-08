"""
Phase 14 — Same-Company Multiple Roles

Goal:
    Validate that multiple roles at the same company are segmented
    into separate experience episodes when boundary evidence supports
    a new role.

Important:
    Same company != same experience.

Architecture under test:

    DocumentBlock[]
        ↓
    MultiBlockCandidateBuilder
        ↓
    ExperienceCandidate[]
        ↓
    ExperienceSegmenter
        ↓
    SegmentationResult
"""

from src.inference.candidate_builder import MultiBlockCandidateBuilder
from src.inference.segmentation import ExperienceSegmenter

from tests.test_multiblock_candidate_builder import block


def make_blocks(*texts):
    """Create deterministic sequential DocumentBlock fixtures."""
    return [
        block(f"b{i}", text, i)
        for i, text in enumerate(texts)
    ]


def build_candidates(blocks):
    """Build ExperienceCandidate objects using the canonical builder."""
    builder = MultiBlockCandidateBuilder()
    return builder.build_from_blocks(blocks)


def segment_blocks(blocks):
    """Build candidates and pass them to the canonical segmenter."""
    candidates = build_candidates(blocks)

    segmenter = ExperienceSegmenter()

    result = segmenter.segment(candidates)

    return candidates, result


# ---------------------------------------------------------------------------
# 1. Same company + different roles + different dates
# ---------------------------------------------------------------------------

def test_same_company_different_roles_different_dates_are_separate():
    blocks = make_blocks(
        "Data Analyst",
        "ABC Technologies",
        "Jan 2022 - Dec 2022",
        "Machine Learning Engineer",
        "ABC Technologies",
        "Jan 2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) >= 2
    assert len(result.segments) == 2


# ---------------------------------------------------------------------------
# 2. Same company + different roles + missing dates
# ---------------------------------------------------------------------------

def test_same_company_different_roles_without_dates_are_separate():
    blocks = make_blocks(
        "Data Analyst",
        "ABC Technologies",
        "Machine Learning Engineer",
        "ABC Technologies",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) >= 2
    assert len(result.segments) == 2


# ---------------------------------------------------------------------------
# 3. Same company + role continuation split across blocks
# ---------------------------------------------------------------------------

def test_role_continuation_at_same_company_does_not_create_false_boundary():
    blocks = make_blocks(
        "Senior Machine Learning",
        "Engineer",
        "ABC Technologies",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    # The continuation must remain part of the same candidate.
    assert len(candidates) == 1

    # More importantly, it must remain one experience.
    assert len(result.segments) == 1

    # Both role-fragment blocks must be preserved as source evidence.
    source_blocks = candidates[0].evidence.metadata.get(
        "source_blocks",
        [],
    )

    assert "b0" in source_blocks
    assert "b1" in source_blocks


# ---------------------------------------------------------------------------
# 4. Repeated company omitted from second role
# ---------------------------------------------------------------------------

def test_same_company_can_be_inherited_for_second_role_without_merging():
    blocks = make_blocks(
        "Data Analyst",
        "ABC Technologies",
        "2021 - 2022",
        "Senior Data Analyst",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) >= 2
    assert len(result.segments) == 2


# ---------------------------------------------------------------------------
# 5. Same company + different locations
# ---------------------------------------------------------------------------

def test_location_change_at_same_company_does_not_by_itself_define_boundary():
    blocks = make_blocks(
        "Data Analyst",
        "ABC Technologies",
        "Pune, India",
        "2021 - 2022",
        "Senior Data Analyst",
        "ABC Technologies",
        "Mumbai, India",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) >= 2
    assert len(result.segments) == 2


# ---------------------------------------------------------------------------
# 6. Same company + current second role
# ---------------------------------------------------------------------------

def test_same_company_current_second_role_is_separate():
    blocks = make_blocks(
        "Data Scientist",
        "ABC Technologies",
        "2020 - 2023",
        "Senior Data Scientist",
        "ABC Technologies",
        "2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) >= 2
    assert len(result.segments) == 2

    assert any(candidate.is_current for candidate in candidates)


# ---------------------------------------------------------------------------
# 7. Description between roles
# ---------------------------------------------------------------------------

def test_description_between_same_company_roles_does_not_merge_roles():
    blocks = make_blocks(
        "Data Analyst",
        "ABC Technologies",
        "2021 - 2022",
        "Built automated reporting dashboards.",
        "Machine Learning Engineer",
        "ABC Technologies",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) >= 2
    assert len(result.segments) == 2


# ---------------------------------------------------------------------------
# 8. Weak / ambiguous role wording
# ---------------------------------------------------------------------------

def test_weak_role_wording_requires_role_evidence_to_split():
    blocks = make_blocks(
        "ABC Technologies",
        "2021 - 2022",
        "Worked with machine learning models",
        "ABC Technologies",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    # Repeated company/date information alone must not force
    # an arbitrary experience split.
    assert len(result.segments) <= 2

    # No fabricated role should be required.
    for candidate in candidates:
        if candidate.role:
            assert len(candidate.role.strip()) > 0


# ---------------------------------------------------------------------------
# 9. False role-like prose
# ---------------------------------------------------------------------------

def test_role_like_prose_does_not_create_spurious_same_company_role():
    blocks = make_blocks(
        "Data Analyst",
        "ABC Technologies",
        "2021 - 2022",
        "Worked across multiple levels of reporting.",
        "Collaborated with engineering teams.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(result.segments) == 1

    assert all(
        candidate.role != "multiple levels"
        for candidate in candidates
    )


# ---------------------------------------------------------------------------
# 10. Three roles at the same company
# ---------------------------------------------------------------------------

def test_three_roles_at_same_company_remain_three_experiences():
    blocks = make_blocks(
        "Data Analyst",
        "ABC Technologies",
        "2020 - 2021",
        "Senior Data Analyst",
        "ABC Technologies",
        "2021 - 2022",
        "Machine Learning Engineer",
        "ABC Technologies",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) >= 3
    assert len(result.segments) == 3