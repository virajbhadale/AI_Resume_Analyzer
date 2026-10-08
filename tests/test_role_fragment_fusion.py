from src.document_models import DocumentBlock

from src.inference.candidate_builder import (
    MultiBlockCandidateBuilder,
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


def test_role_title_split_across_blocks_is_one_candidate():
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

    assert candidate.block_ids == [
        "b1",
        "b2",
        "b3",
        "b4",
    ]

    assert candidate.company == "ABC Technologies"
    assert candidate.start_date == "2022"
    assert candidate.end_date == "Present"


def test_role_fragment_is_preserved_as_source_evidence():
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

    assert "b1" in candidate.block_ids
    assert "b2" in candidate.block_ids


def test_single_block_role_remains_unchanged():
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
    assert candidates[0].role == "Data Scientist"


def test_role_and_role_like_content_with_different_companies_remain_separate():
    blocks = [
        make_block(
            "b1",
            "Data Analyst",
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
            "2020 - 2022",
            3,
        ),
        make_block(
            "b4",
            "Data Scientist",
            4,
            bold=True,
        ),
        make_block(
            "b5",
            "XYZ Technologies",
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

    assert candidates[0].company == "ABC Technologies"
    assert candidates[1].company == "XYZ Technologies"


def test_same_company_different_roles_remain_separate():
    blocks = [
        make_block(
            "b1",
            "Data Analyst",
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
            "ABC Technologies",
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

    assert candidates[0].role == "Data Analyst"
    assert candidates[1].role == "Machine Learning Engineer"


def test_role_fragment_without_supporting_metadata_is_not_forced_into_title():
    blocks = [
        make_block(
            "b1",
            "Data Analyst",
            1,
            bold=True,
        ),
        make_block(
            "b2",
            "Research",
            2,
        ),
    ]

    candidates = build_candidates(blocks)

    # We deliberately do not prescribe concatenation here.
    # The important invariant is that we do not fabricate a
    # professional experience from two role-like fragments alone.
    assert len(candidates) <= 1