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


def test_location_belongs_to_nearest_experience():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),
        make_block("b4", "Pune, India", 4),

        make_block("b5", "Machine Learning Engineer", 5, bold=True),
        make_block("b6", "Company B", 6),
        make_block("b7", "2023 - Present", 7),
        make_block("b8", "Bangalore, India", 8),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 2

    assert candidates[0].company == "Company A"
    assert candidates[0].location == "Pune, India"

    assert candidates[1].company == "Company B"
    assert candidates[1].location == "Bangalore, India"


def test_date_belongs_to_nearest_experience():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),

        make_block("b4", "Machine Learning Engineer", 4, bold=True),
        make_block("b5", "Company B", 5),
        make_block("b6", "2023 - Present", 6),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 2

    assert candidates[0].company == "Company A"
    assert candidates[0].start_date is not None

    assert candidates[1].company == "Company B"
    assert candidates[1].start_date is not None

    assert candidates[0].start_date != candidates[1].start_date


def test_missing_location_is_not_filled_from_previous_experience():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),
        make_block("b4", "Pune, India", 4),

        make_block("b5", "Machine Learning Engineer", 5, bold=True),
        make_block("b6", "Company B", 6),
        make_block("b7", "2023 - Present", 7),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 2

    assert candidates[0].location == "Pune, India"
    assert candidates[1].location is None


def test_missing_date_is_not_filled_from_previous_experience():
    blocks = [
        make_block("b1", "Data Analyst", 1, bold=True),
        make_block("b2", "Company A", 2),
        make_block("b3", "2020 - 2022", 3),

        make_block("b4", "Machine Learning Engineer", 4, bold=True),
        make_block("b5", "Company B", 5),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 2

    assert candidates[0].start_date is not None
    assert candidates[1].start_date is None