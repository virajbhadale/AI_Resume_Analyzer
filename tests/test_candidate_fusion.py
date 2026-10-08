from src.document_models import DocumentBlock
from src.inference.candidate_builder import MultiBlockCandidateBuilder


def block(
    block_id,
    text,
    reading_order,
    *,
    bold=False,
    y=0,
    section=None,
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


def test_complete_experience_candidate_preserves_all_fields():
    blocks = [
        block(
            "b1",
            "Machine Learning Engineer",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "OpenAI",
            2,
            y=125,
        ),
        block(
            "b3",
            "Jan 2024 - Present",
            3,
            y=150,
        ),
        block(
            "b4",
            "San Francisco, CA",
            4,
            y=175,
        ),
        block(
            "b5",
            "Built production inference systems.",
            5,
            y=200,
        ),
    ]

    builder = MultiBlockCandidateBuilder()

    evidence = builder.build_block_evidence(blocks)

    print(
        "\nLOCATION EVIDENCE:",
        [
            {
                "block_id": item.block_id,
                "text": item.text,
                "location": item.location,
                "location_confidence": item.location_confidence,
                "has_location": item.has_location,
            }
            for item in evidence
        ],
    )

    candidates = builder.build_from_blocks(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.role == "Machine Learning Engineer"
    assert candidate.company == "OpenAI"
    assert candidate.location == "San Francisco, CA"
    assert candidate.start_date == "Jan 2024"
    assert candidate.is_current is True
    assert "b1" in candidate.block_ids
    assert "b2" in candidate.block_ids
    assert "b3" in candidate.block_ids


def test_missing_company_remains_none():
    blocks = [
        block(
            "b1",
            "Data Analyst Intern",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Pune, India",
            2,
            y=125,
        ),
        block(
            "b3",
            "Jun 2023 - Aug 2023",
            3,
            y=150,
        ),
    ]

    builder = MultiBlockCandidateBuilder()

    evidence = builder.build_block_evidence(blocks)

    print(
        "\nLOCATION EVIDENCE:",
        [
            {
                "block_id": item.block_id,
                "text": item.text,
                "location": item.location,
                "location_confidence": item.location_confidence,
                "has_location": item.has_location,
            }
            for item in evidence
        ],
    )

    candidates = builder.build_from_blocks(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.role == "Data Analyst Intern"
    assert candidate.company is None
    assert candidate.location == "Pune, India"
    assert candidate.start_date == "Jun 2023"
    assert candidate.end_date == "Aug 2023"


def test_different_companies_do_not_cross_assign_dates():
    blocks = [
        block(
            "b1",
            "Data Analyst",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Company A",
            2,
            y=125,
        ),
        block(
            "b3",
            "2020 - 2022",
            3,
            y=150,
        ),
        block(
            "b4",
            "ML Engineer",
            4,
            bold=True,
            y=220,
        ),
        block(
            "b5",
            "Company B",
            5,
            y=245,
        ),
        block(
            "b6",
            "2023 - Present",
            6,
            y=270,
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 2

    assert candidates[0].company == "Company A"
    assert candidates[1].company == "Company B"

    assert candidates[0].end_date != candidates[1].end_date
    assert candidates[0].end_date == "2022"
    assert candidates[1].is_current is True


def test_role_company_date_are_fused_from_multiple_blocks():
    blocks = [
        block(
            "b1",
            "Data Scientist",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Acme AI Labs",
            2,
            y=125,
        ),
        block(
            "b3",
            "Jan 2022 - Dec 2023",
            3,
            y=150,
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.role == "Data Scientist"
    assert candidate.company == "Acme AI Labs"
    assert candidate.start_date == "Jan 2022"
    assert candidate.end_date == "Dec 2023"


def test_candidate_does_not_require_company():
    blocks = [
        block(
            "b1",
            "Research Intern",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Jun 2022 - Aug 2022",
            2,
            y=125,
        ),
        block(
            "b3",
            "Conducted experiments and analyzed results.",
            3,
            y=150,
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.role == "Research Intern"
    assert candidate.company is None
    assert candidate.start_date == "Jun 2022"
    assert candidate.end_date == "Aug 2022"


def test_candidate_preserves_missing_date():
    blocks = [
        block(
            "b1",
            "Software Engineer",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Example Technologies",
            2,
            y=125,
        ),
        block(
            "b3",
            "Developed backend services using Python.",
            3,
            y=150,
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.role == "Software Engineer"
    assert candidate.company == "Example Technologies"
    assert candidate.start_date is None
    assert candidate.end_date is None


def test_project_section_is_not_fused_into_experience():
    blocks = [
        block(
            "b1",
            "Software Engineer",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Acme Corp",
            2,
            y=125,
        ),
        block(
            "b3",
            "2024 - Present",
            3,
            y=150,
        ),
        block(
            "b4",
            "Projects",
            4,
            section="projects",
            y=200,
        ),
        block(
            "b5",
            "Resume Analyzer",
            5,
            section="projects",
            y=225,
        ),
    ]

    candidates = MultiBlockCandidateBuilder().build_from_blocks(blocks)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert "b4" not in candidate.block_ids
    assert "b5" not in candidate.block_ids


def test_second_experience_window_does_not_cross_previous_role():
    blocks = [
        block(
            "b1",
            "Data Analyst",
            1,
            bold=True,
            y=100,
        ),
        block(
            "b2",
            "Company A",
            2,
            y=125,
        ),
        block(
            "b3",
            "2020 - 2022",
            3,
            y=150,
        ),
        block(
            "b4",
            "ML Engineer",
            4,
            bold=True,
            y=220,
        ),
        block(
            "b5",
            "Company B",
            5,
            y=245,
        ),
        block(
            "b6",
            "2023 - Present",
            6,
            y=270,
        ),
    ]

    builder = MultiBlockCandidateBuilder()

    evidence = builder.build_block_evidence(blocks)

    window = builder._candidate_window(
        evidence,
        3,
    )

    assert [item.block_id for item in window] == [
        "b4",
        "b5",
        "b6",
    ]