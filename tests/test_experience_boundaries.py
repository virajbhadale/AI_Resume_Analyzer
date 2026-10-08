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


def test_two_different_roles_and_companies_create_two_candidates():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),
        make_block("b4", "Built reporting pipelines.", 4),

        make_block("b5", "Machine Learning Engineer", 5, bold=True),
        make_block("b6", "Company B", 6),
        make_block("b7", "2023 - Present", 7),
        make_block("b8", "Built ML systems.", 8),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2

    assert candidates[0].role == "Data Analyst"
    assert candidates[0].company == "Company A"

    assert candidates[1].role == "Machine Learning Engineer"
    assert candidates[1].company == "Company B"


def test_continuation_bullets_do_not_create_extra_candidate():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),
        make_block("b4", "- Built reporting pipelines.", 4),
        make_block("b5", "- Automated monthly reports.", 5),
        make_block("b6", "- Improved dashboard accuracy.", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1
    assert candidates[0].role == "Data Analyst"
    assert candidates[0].company == "Company A"


def test_different_companies_do_not_merge():
    blocks = [
        make_block("b1", "Software Engineer", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),

        make_block("b4", "Software Engineer", 4, bold=True),
        make_block("b5", "Company B", 5),
        make_block("b6", "2023 - Present", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2
    assert candidates[0].company == "Company A"
    assert candidates[1].company == "Company B"


def test_same_company_different_roles_remain_separate():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "ABC Technologies", 2),
        make_block("b3", "2020 - 2022", 3),

        make_block("b4", "Senior Data Analyst", 4, bold=True),
        make_block("b5", "ABC Technologies", 5),
        make_block("b6", "2022 - Present", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2

    assert candidates[0].role == "Data Analyst"
    assert candidates[1].role == "Senior Data Analyst"

    assert candidates[0].company == "ABC Technologies"
    assert candidates[1].company == "ABC Technologies"


def test_new_role_with_new_date_creates_new_candidate_even_without_company():
    blocks = [
        make_block("b1", "Research Intern", 1, bold=True),
        make_block("b2", "2021 - 2022", 2),
        make_block("b3", "Performed research experiments.", 3),

        make_block("b4", "Research Assistant", 4, bold=True),
        make_block("b5", "2023 - Present", 5),
        make_block("b6", "Supported research projects.", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2

    assert candidates[0].role == "Research Intern"
    assert candidates[1].role == "Research Assistant"


def test_description_between_roles_does_not_become_boundary_by_itself():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),
        make_block(
            "b4",
            "Developed automated reports and dashboards.",
            4,
        ),

        make_block("b5", "Machine Learning Engineer", 5, bold=True),
        make_block("b6", "Company B", 6),
        make_block("b7", "2023 - Present", 7),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 2

    assert candidates[0].role == "Data Analyst"
    assert candidates[1].role == "Machine Learning Engineer"


def test_location_change_alone_does_not_create_new_candidate():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),
        make_block("b4", "Pune, India", 4),
        make_block("b5", "Mumbai, India", 5),
        make_block("b6", "- Worked with multiple teams.", 6),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) == 1
    assert candidates[0].role == "Data Analyst"
    assert candidates[0].company == "Company A"