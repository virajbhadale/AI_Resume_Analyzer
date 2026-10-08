from src.document_models import DocumentBlock
from src.inference.candidate_builder import MultiBlockCandidateBuilder


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
    return MultiBlockCandidateBuilder().build_from_blocks(blocks)


def test_same_company_internship_to_full_time_is_separate():
    blocks = [
        make_block("b1", "Software Engineer Intern", 1, bold=True),
        make_block("b2", "ABC Technologies", 2),
        make_block("b3", "2021 - 2022", 3),

        make_block("b4", "Software Engineer", 4, bold=True),
        make_block("b5", "ABC Technologies", 5),
        make_block("b6", "2022 - Present", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2
    assert candidates[0].role == "Software Engineer Intern"
    assert candidates[1].role == "Software Engineer"


def test_same_role_different_companies_is_separate():
    blocks = [
        make_block("b1", "Data Scientist", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),

        make_block("b4", "Data Scientist", 4, bold=True),
        make_block("b5", "Company B", 5),
        make_block("b6", "2022 - Present", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2
    assert candidates[0].company == "Company A"
    assert candidates[1].company == "Company B"


def test_repeated_company_header_does_not_merge_different_roles():
    blocks = [
        make_block("b1", "ABC Technologies", 1, bold=True),
        make_block("b2", "Data Analyst", 2),
        make_block("b3", "2020 - 2022", 3),

        make_block("b4", "ABC Technologies", 4, bold=True),
        make_block("b5", "Senior Data Analyst", 5),
        make_block("b6", "2022 - Present", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2
    assert candidates[0].role == "Data Analyst"
    assert candidates[1].role == "Senior Data Analyst"


def test_role_continuation_does_not_create_new_candidate():
    blocks = [
        make_block("b1", "Senior Machine Learning", 1, bold=True),
        make_block("b2", "Engineer", 2),
        make_block("b3", "ABC Technologies", 3),
        make_block("b4", "2022 - Present", 4),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1


def test_date_only_after_existing_experience_does_not_create_candidate():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "ABC Technologies", 2),
        make_block("b3", "2020 - 2022", 3),
        make_block("b4", "2022 - 2023", 4),
        make_block("b5", "- Continued analytical responsibilities.", 5),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1


def test_company_only_after_completed_experience_requires_role_context():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),

        make_block("b4", "Company B", 4),
        make_block("b5", "2023 - Present", 5),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) >= 1
    assert candidates[0].company == "Company A"


def test_research_roles_can_be_separate_experiences():
    blocks = [
        make_block("b1", "Research Intern", 1, bold=True),
        make_block("b2", "University Lab", 2),
        make_block("b3", "2021 - 2022", 3),

        make_block("b4", "Research Assistant", 4, bold=True),
        make_block("b5", "University Lab", 5),
        make_block("b6", "2022 - Present", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2
    assert candidates[0].role == "Research Intern"
    assert candidates[1].role == "Research Assistant"


def test_current_role_without_end_date_remains_valid():
    blocks = [
        make_block("b1", "Machine Learning Engineer", 1, bold=True),
        make_block("b2", "ABC Technologies", 2),
        make_block("b3", "2024 - Present", 3),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1
    assert candidates[0].role == "Machine Learning Engineer"
    assert candidates[0].company == "ABC Technologies"
    assert candidates[0].start_date is not None