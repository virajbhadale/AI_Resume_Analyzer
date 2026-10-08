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


def test_sentence_like_bullet_does_not_become_experience():
    blocks = [
        make_block(
            "b1",
            "- Presented findings at weekly research meetings",
            1,
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert candidates == []


def test_ordinary_prose_does_not_become_experience():
    blocks = [
        make_block(
            "b1",
            "Worked with various teams to improve reporting processes.",
            1,
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert candidates == []


def test_project_content_does_not_become_experience():
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
            "Developed a forecasting model using Python.",
            2,
            section="projects",
        ),
        make_block(
            "b3",
            "2023",
            3,
            section="projects",
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert candidates == []


def test_skills_content_does_not_become_experience():
    blocks = [
        make_block(
            "b1",
            "Python",
            1,
            section="skills",
        ),
        make_block(
            "b2",
            "Machine Learning",
            2,
            section="skills",
        ),
        make_block(
            "b3",
            "TensorFlow",
            3,
            section="skills",
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert candidates == []


def test_real_experience_can_exist_next_to_negative_content():
    blocks = [
        make_block(
            "b1",
            "- Presented findings at weekly research meetings",
            1,
        ),
        make_block(
            "b2",
            "Data Analyst",
            2,
            bold=True,
        ),
        make_block(
            "b3",
            "ABC Technologies",
            3,
        ),
        make_block(
            "b4",
            "2022 - 2024",
            4,
        ),
        make_block(
            "b5",
            "Built automated reporting pipelines.",
            5,
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 1
    assert candidates[0].role == "Data Analyst"
    assert candidates[0].company == "ABC Technologies"