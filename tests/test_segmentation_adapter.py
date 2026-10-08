from src.inference.anchors import AnchorDetector
from src.inference.candidate_builder import CandidateBuilder
from src.inference.segmentation_adapter import (
    ExperienceSegmentationAdapter,
)
from src.inference.segmentation import (
    ExperienceSegmenter,
)


def test_adapter_detects_complete_experience_anchor(
    make_document_block,
):
    block = make_document_block(
        text=(
            "Data Analyst Intern\n"
            "ComplyHappy Pvt Ltd\n"
            "Pune, India\n"
            "19 Jun 2023 - 11 Aug 2023"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["anchor_count"] == 1
    assert result.metadata["candidate_count"] == 1
    assert result.metadata["segment_count"] == 1

    segment = result.segments[0]

    assert block.id in segment.block_ids
    assert block.id in segment.anchor_block_ids


def test_adapter_separates_different_roles(
    make_document_block,
):
    block_1 = make_document_block(
        text=(
            "Data Analyst\n"
            "ComplyHappy Pvt Ltd\n"
            "2023 - 2023"
        ),
        section="experience",
    )

    block_2 = make_document_block(
        text=(
            "AI Engineer Intern\n"
            "Marva AI\n"
            "2025 - 2025"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze(
        [block_1, block_2]
    )

    assert result.metadata["candidate_count"] == 2
    assert result.metadata["segment_count"] == 2


def test_adapter_does_not_process_non_experience_sections(
    make_document_block,
):
    block = make_document_block(
        text=(
            "Data Analyst\n"
            "Google\n"
            "2023 - 2024"
        ),
        section="education",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["anchor_count"] == 0
    assert result.metadata["candidate_count"] == 0
    assert result.metadata["segment_count"] == 0


def test_adapter_preserves_candidate_identity(
    make_document_block,
):
    block = make_document_block(
        text=(
            "Software Engineer\n"
            "Google\n"
            "2024 - Present"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    anchors = adapter.detect_anchors([block])
    candidates = adapter.build_candidates(anchors)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.candidate_id.startswith(
        "experience_candidate_"
    )

    assert block.id in candidate.block_ids
    assert candidate.role == "Software Engineer"
    assert candidate.company == "Google"