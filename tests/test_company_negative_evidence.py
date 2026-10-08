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


def test_sentence_like_bullet_has_no_company_evidence():
    block = make_block(
        "b1",
        "- Presented findings at weekly research meetings",
        1,
    )

    evidence = MultiBlockCandidateBuilder().build_block_evidence([block])[0]

    assert evidence.company is None


def test_sentence_like_prose_has_no_company_evidence():
    block = make_block(
        "b1",
        "Worked with various teams to improve reporting processes.",
        1,
    )

    evidence = MultiBlockCandidateBuilder().build_block_evidence([block])[0]

    assert evidence.company is None


def test_real_company_still_produces_company_evidence():
    block = make_block(
        "b1",
        "ABC Technologies",
        1,
    )

    evidence = MultiBlockCandidateBuilder().build_block_evidence([block])[0]

    assert evidence.company == "ABC Technologies"


def test_negative_bullet_does_not_produce_company_evidence_before_real_experience():
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
    ]

    evidence = MultiBlockCandidateBuilder().build_block_evidence(blocks)

    assert evidence[0].company is None
    assert evidence[2].company == "ABC Technologies"