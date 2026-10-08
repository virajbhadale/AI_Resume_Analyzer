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

    segmenter = ExperienceSegmenter()
    result = segmenter.segment(candidates)

    return candidates, result


def test_same_role_different_companies_creates_two_experiences():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2022 - 2023",
        "Built forecasting models.",
        "Data Scientist",
        "Company B",
        "2023 - Present",
        "Built machine learning pipelines.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_same_role_different_companies_without_dates():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "Built forecasting models.",
        "Data Scientist",
        "Company B",
        "Built machine learning pipelines.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_same_role_company_transition_across_blocks():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2022 - 2023",
        "Data Scientist",
        "Company B",
        "2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_same_role_second_company_explicitly_separates_experience():
    blocks = make_blocks(
        "Machine Learning Engineer",
        "OpenAI",
        "2022 - 2023",
        "Machine Learning Engineer",
        "Google",
        "2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_same_role_different_locations_and_companies():
    blocks = make_blocks(
        "Data Analyst",
        "Company A",
        "Pune, India",
        "2022 - 2023",
        "Data Analyst",
        "Company B",
        "Mumbai, India",
        "2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_same_role_second_company_current():
    blocks = make_blocks(
        "AI Engineer",
        "Company A",
        "2021 - 2023",
        "AI Engineer",
        "Company B",
        "2023 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_description_between_same_role_experiences():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2021 - 2022",
        "Developed predictive models.",
        "Data Scientist",
        "Company B",
        "2022 - Present",
        "Developed production ML systems.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 2
    assert len(result.segments) == 2


def test_same_role_same_company_continuation_stays_one_experience():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2022 - Present",
        "Developed forecasting systems.",
        "Improved model performance.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_company_like_prose_does_not_create_false_experience():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2022 - Present",
        "Worked with Google and Microsoft technologies.",
        "Presented findings at weekly research meetings.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1


def test_three_companies_same_role_create_three_experiences():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2020 - 2021",
        "Data Scientist",
        "Company B",
        "2021 - 2022",
        "Data Scientist",
        "Company C",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 3
    assert len(result.segments) == 3


def test_same_role_different_companies_preserves_boundary_evidence():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2021 - 2022",
        "Data Scientist",
        "Company B",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(result.segments) == 2
    assert len(result.boundaries) == 1

    boundary = result.boundaries[0]

    assert boundary.supporting_evidence
    assert any(
        signal.metadata.get("boundary_signal") == "company_change"
        for signal in boundary.supporting_evidence
    )


def test_same_role_different_companies_have_distinct_source_blocks():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2021 - 2022",
        "Data Scientist",
        "Company B",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(result.segments) == 2

    first_source = candidates[0].evidence.metadata.get(
        "source_blocks",
        [],
    )
    second_source = candidates[1].evidence.metadata.get(
        "source_blocks",
        [],
    )

    assert first_source
    assert second_source
    assert set(first_source).isdisjoint(second_source)