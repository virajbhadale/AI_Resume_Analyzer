from src.inference.candidate_builder import MultiBlockCandidateBuilder
from src.inference.segmentation import ExperienceSegmenter

from tests.test_multiblock_candidate_builder import block


def make_blocks(*texts):
    return [
        block(f"b{i}", text, i)
        for i, text in enumerate(texts)
    ]


def build_candidates(blocks):
    builder = MultiBlockCandidateBuilder()
    return builder.build_from_blocks(blocks)


def segment_blocks(blocks):
    candidates = build_candidates(blocks)
    result = ExperienceSegmenter().segment(candidates)
    return candidates, result


def get_dates(candidate):
    return (
        candidate.start_date,
        candidate.end_date,
    )


def test_two_experiences_keep_distinct_date_ranges():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "Jan 2021 - Dec 2022",
        "Built forecasting models.",
        "Data Scientist",
        "Company B",
        "Jan 2023 - Present",
        "Built production ML systems.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2

    first_dates = get_dates(candidates[0])
    second_dates = get_dates(candidates[1])

    assert first_dates != second_dates


def test_same_company_different_roles_keep_their_dates():
    blocks = make_blocks(
        "Data Analyst",
        "ABC Technologies",
        "Jan 2021 - Dec 2022",
        "Machine Learning Engineer",
        "ABC Technologies",
        "Jan 2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2

    assert get_dates(candidates[0]) != get_dates(candidates[1])


def test_same_role_different_companies_keep_their_dates():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2020 - 2022",
        "Data Scientist",
        "Company B",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2

    assert get_dates(candidates[0]) != get_dates(candidates[1])


def test_date_before_role_can_be_associated_with_following_experience():
    blocks = make_blocks(
        "Jan 2021 - Dec 2022",
        "Data Scientist",
        "Company A",
        "Built forecasting models.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    start, end = get_dates(candidates[0])

    assert start is not None
    assert end is not None


def test_date_after_company_can_be_associated_with_experience():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "Jan 2021 - Dec 2022",
        "Built forecasting models.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    start, end = get_dates(candidates[0])

    assert start is not None
    assert end is not None


def test_current_experience_preserves_present_date():
    blocks = make_blocks(
        "Machine Learning Engineer",
        "Company A",
        "2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    start, end = get_dates(candidates[0])

    assert start is not None
    assert end is not None


def test_missing_date_does_not_create_false_date_evidence():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "Built forecasting models.",
        "Data Scientist",
        "Company B",
        "Built ML systems.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_description_date_like_text_does_not_become_second_experience_date():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2022 - Present",
        "Improved performance by 20% during 2023.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_year_only_date_range_is_preserved():
    blocks = make_blocks(
        "Data Analyst",
        "Company A",
        "2021 - 2023",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    start, end = get_dates(candidates[0])

    assert start is not None
    assert end is not None


def test_month_year_date_range_is_preserved():
    blocks = make_blocks(
        "Data Analyst",
        "Company A",
        "Jun 2022 - Aug 2024",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    start, end = get_dates(candidates[0])

    assert start is not None
    assert end is not None


def test_three_experiences_keep_three_temporal_assignments():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2019 - 2020",
        "Data Scientist",
        "Company B",
        "2020 - 2022",
        "Data Scientist",
        "Company C",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 3
    assert len(result.segments) == 3

    dates = [get_dates(candidate) for candidate in candidates]

    assert len(set(dates)) == 3


def test_distinct_date_ranges_create_boundary_evidence():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2020 - 2022",
        "Data Scientist",
        "Company B",
        "2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1

    boundary = result.boundaries[0]

    assert any(
        signal.metadata.get("boundary_signal") == "date_range_change"
        for signal in boundary.supporting_evidence
    )