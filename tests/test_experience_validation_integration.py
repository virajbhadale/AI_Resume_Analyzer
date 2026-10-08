from src.document_models import DocumentBlock

from src.inference.candidate_builder import (
    MultiBlockCandidateBuilder,
)
from src.inference.candidates import (
    CandidateStatus,
)
from src.inference.validation import (
    ExperienceCandidateValidator,
)


def make_block(
    block_id,
    text,
    reading_order,
    *,
    bold=False,
    y=0,
    section="experience",
):
    return DocumentBlock(
        id=block_id,
        text=text,
        page_number=1,
        x=0,
        y=y,
        width=500,
        height=20,
        reading_order=reading_order,
        is_bold=bold,
        section=section,
    )


def build_candidates(blocks):
    return MultiBlockCandidateBuilder().build_from_blocks(
        blocks
    )


def validate_candidates(candidates):
    validator = ExperienceCandidateValidator()

    return [
        validator.validate(candidate)
        for candidate in candidates
    ]


def test_real_role_company_date_candidate_is_validated():
    blocks = [
        make_block(
            "b1",
            "Data Scientist",
            1,
            bold=True,
        ),
        make_block(
            "b2",
            "ABC Technologies",
            2,
        ),
        make_block(
            "b3",
            "2022 - 2024",
            3,
        ),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1

    results = validate_candidates(candidates)

    assert len(results) == 1
    assert results[0].status == CandidateStatus.VALIDATED
    assert results[0].is_valid is True


def test_missing_company_candidate_can_still_be_validated():
    blocks = [
        make_block(
            "b1",
            "Data Analyst Intern",
            1,
            bold=True,
        ),
        make_block(
            "b2",
            "Pune, India",
            2,
        ),
        make_block(
            "b3",
            "Jun 2023 - Aug 2023",
            3,
        ),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.role == "Data Analyst Intern"
    assert candidate.company is None
    assert candidate.location == "Pune, India"
    assert candidate.start_date is not None

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.VALIDATED
    assert result.is_valid is True


def test_candidate_evidence_is_generated_before_validation():
    blocks = [
        make_block(
            "b1",
            "Machine Learning Engineer",
            1,
            bold=True,
        ),
        make_block(
            "b2",
            "ABC Technologies",
            2,
        ),
        make_block(
            "b3",
            "2024 - Present",
            3,
        ),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    signal_types = {
        signal.signal_type.value
        for signal in candidate.evidence.signals
    }

    assert "role" in signal_types
    assert "company" in signal_types
    assert "date" in signal_types

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.VALIDATED


def test_role_continuation_remains_one_validated_candidate():
    blocks = [
        make_block(
            "b1",
            "Senior Machine Learning",
            1,
            bold=True,
        ),
        make_block(
            "b2",
            "Engineer",
            2,
        ),
        make_block(
            "b3",
            "ABC Technologies",
            3,
        ),
        make_block(
            "b4",
            "2022 - Present",
            4,
        ),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    result = ExperienceCandidateValidator().validate(
        candidate
    )

    assert result.status == CandidateStatus.VALIDATED


def test_two_experiences_are_validated_independently():
    blocks = [
        make_block(
            "b1",
            "Data Analyst",
            1,
            bold=True,
        ),
        make_block(
            "b2",
            "Company A",
            2,
        ),
        make_block(
            "b3",
            "2020 - 2022",
            3,
        ),
        make_block(
            "b4",
            "Machine Learning Engineer",
            4,
            bold=True,
        ),
        make_block(
            "b5",
            "Company B",
            5,
        ),
        make_block(
            "b6",
            "2023 - Present",
            6,
        ),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2

    results = validate_candidates(candidates)

    assert len(results) == 2
    assert all(
        result.status == CandidateStatus.VALIDATED
        for result in results
    )


def test_project_section_is_not_treated_as_experience_candidate():
    blocks = [
        make_block(
            "b1",
            "Machine Learning Project",
            1,
            bold=True,
            section="projects",
        ),
        make_block(
            "b2",
            "ABC Technologies",
            2,
            section="projects",
        ),
        make_block(
            "b3",
            "2024",
            3,
            section="projects",
        ),
    ]

    candidates = build_candidates(blocks)

    assert candidates == []


def test_negative_sentence_does_not_create_validated_experience():
    blocks = [
        make_block(
            "b1",
            "- Presented findings at weekly research meetings",
            1,
        ),
    ]

    candidates = build_candidates(blocks)

    assert candidates == []


def test_validation_preserves_candidate_provenance():
    blocks = [
        make_block(
            "b1",
            "Data Scientist",
            1,
            bold=True,
        ),
        make_block(
            "b2",
            "ABC Technologies",
            2,
        ),
        make_block(
            "b3",
            "2022 - 2024",
            3,
        ),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    original_block_ids = list(
        candidate.block_ids
    )

    original_field_sources = dict(
        candidate.evidence.metadata[
            "field_sources"
        ]
    )

    ExperienceCandidateValidator().validate(
        candidate
    )

    assert candidate.block_ids == original_block_ids

    assert (
        candidate.evidence.metadata[
            "field_sources"
        ]
        == original_field_sources
    )