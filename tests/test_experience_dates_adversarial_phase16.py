from src.inference.candidate_builder import MultiBlockCandidateBuilder
from src.inference.segmentation import ExperienceSegmenter

from tests.test_multiblock_candidate_builder import block


def make_blocks(*texts):
    return [
        block(f"b{i}", text, i)
        for i, text in enumerate(texts)
    ]


def build_candidates(blocks):
    return MultiBlockCandidateBuilder().build_from_blocks(blocks)


def segment_blocks(blocks):
    candidates = build_candidates(blocks)
    result = ExperienceSegmenter().segment(candidates)
    return candidates, result


def test_date_like_metric_in_description_does_not_create_experience():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2022 - Present",
        "Improved model accuracy from 80% to 95% in 2023.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_date_like_project_year_does_not_create_second_experience():
    blocks = make_blocks(
        "Machine Learning Engineer",
        "Company A",
        "2021 - Present",
        "Project launched in 2023.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_date_inside_bullet_does_not_override_experience_date():
    blocks = make_blocks(
        "Data Analyst",
        "Company A",
        "2022 - Present",
        "Presented results on 15/10/2023.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_multiple_date_mentions_in_description_do_not_create_multiple_experiences():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2021 - Present",
        "Worked on projects from 2022 to 2023.",
        "Presented results in 2024.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_education_like_dates_do_not_create_false_work_experience():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2022 - Present",
        "B.Sc. Data Science, 2019 - 2022.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_certification_date_does_not_create_false_experience():
    blocks = make_blocks(
        "Machine Learning Engineer",
        "Company A",
        "2022 - Present",
        "AWS Certified Solutions Architect - 2023.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_reversed_date_range_does_not_create_extra_experience():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2023 - 2021",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_isolated_date_without_role_or_company_does_not_create_experience():
    blocks = make_blocks(
        "2021 - 2022",
        "Built several machine learning systems.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 0
    assert len(result.segments) == 0


def test_missing_start_date_does_not_create_false_boundary():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "Present",
        "Data Scientist",
        "Company B",
        "2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) >= 1
    assert len(result.segments) >= 1


def test_standalone_date_does_not_invent_missing_end_date():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "Jan 2022",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    start, end = (
        candidates[0].start_date,
        candidates[0].end_date,
    )

    # Current DateExtractor contract supports date ranges,
    # not a standalone employment start date.
    assert start is None
    assert end is None


def test_current_role_followed_by_older_role_preserves_two_experiences():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2023 - Present",
        "Data Scientist",
        "Company B",
        "2020 - 2023",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_adjacent_experiences_with_only_date_transition_remain_distinct():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2020 - 2022",
        "Data Scientist",
        "Company B",
        "2022 - 2024",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2