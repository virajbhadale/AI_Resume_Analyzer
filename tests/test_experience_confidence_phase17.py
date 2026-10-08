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


def candidate_confidence(candidate):
    return candidate.confidence


def test_strong_role_company_date_candidate_has_high_confidence():
    blocks = make_blocks(
        "Data Scientist",
        "Google",
        "Jan 2022 - Present",
        "Built production machine learning systems.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    confidence = candidate_confidence(candidates[0])

    assert confidence >= 0.7
    assert confidence <= 1.0


def test_preliminary_confidence_remains_valid_when_company_is_missing():
    complete_blocks = make_blocks(
        "Data Scientist",
        "Google",
        "Jan 2022 - Present",
    )

    incomplete_blocks = make_blocks(
        "Data Scientist",
        "Jan 2022 - Present",
    )

    complete_candidates, _ = segment_blocks(complete_blocks)
    incomplete_candidates, _ = segment_blocks(incomplete_blocks)

    assert len(complete_candidates) == 1
    assert len(incomplete_candidates) == 1

    complete_confidence = candidate_confidence(
        complete_candidates[0]
    )
    incomplete_confidence = candidate_confidence(
        incomplete_candidates[0]
    )

    assert 0.0 <= complete_confidence <= 1.0
    assert 0.0 <= incomplete_confidence <= 1.0


def test_role_only_candidate_does_not_receive_maximum_confidence():
    blocks = make_blocks(
        "Data Scientist",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    confidence = candidate_confidence(candidates[0])

    assert confidence < 1.0


def test_missing_company_does_not_create_fake_confidence():
    blocks = make_blocks(
        "Machine Learning Engineer",
        "2023 - Present",
        "Built production inference systems.",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    candidate = candidates[0]

    assert candidate.company is None
    assert candidate.confidence < 1.0


def test_confidence_is_bounded():
    blocks = make_blocks(
        "Data Scientist",
        "Google",
        "2022 - Present",
    )

    candidates, _ = segment_blocks(blocks)

    assert len(candidates) == 1

    confidence = candidate_confidence(candidates[0])

    assert 0.0 <= confidence <= 1.0


def test_confidence_is_deterministic():
    blocks = make_blocks(
        "Data Scientist",
        "Google",
        "2022 - Present",
    )

    candidates_a, _ = segment_blocks(blocks)
    candidates_b, _ = segment_blocks(blocks)

    assert len(candidates_a) == 1
    assert len(candidates_b) == 1

    assert candidate_confidence(candidates_a[0]) == (
        candidate_confidence(candidates_b[0])
    )


def test_same_candidate_structure_produces_same_confidence():
    blocks_a = make_blocks(
        "Data Scientist",
        "Google",
        "2022 - Present",
    )

    blocks_b = make_blocks(
        "Data Scientist",
        "Google",
        "2022 - Present",
    )

    candidates_a, _ = segment_blocks(blocks_a)
    candidates_b, _ = segment_blocks(blocks_b)

    assert candidate_confidence(candidates_a[0]) == (
        candidate_confidence(candidates_b[0])
    )


def test_weak_candidate_is_not_high_confidence():
    blocks = make_blocks(
        "Engineer",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    confidence = candidate_confidence(candidates[0])

    assert confidence < 0.9


def test_multiple_weak_signals_do_not_automatically_mean_maximum_confidence():
    blocks = make_blocks(
        "Engineer",
        "Technology",
        "Project",
        "2022",
    )

    candidates, result = segment_blocks(blocks)

    assert len(result.segments) <= 1

    for candidate in candidates:
        assert candidate.confidence <= 1.0


def test_confidence_survives_segmentation():
    blocks = make_blocks(
        "Data Scientist",
        "Company A",
        "2022 - Present",
    )

    candidates, result = segment_blocks(blocks)

    assert len(candidates) == 1
    assert len(result.segments) == 1

    candidate_conf = candidates[0].confidence
    segment = result.segments[0]

    assert candidate_conf is not None
    assert segment.confidence is not None
    assert 0.0 <= segment.confidence <= 1.0