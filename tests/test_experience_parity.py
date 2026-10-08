from src.document_models import BBox, DocumentBlock
from src.inference.parity import ExperienceParityHarness
from src.extraction.experience import ExperienceExtractor
from src.document_models import DocumentBlock

def make_block(
    index,
    text,
    *,
    section="experience",
    bold=False,
):
    return DocumentBlock(
        id=f"b{index}",
        text=text,
        page=1,
        section=section,
        is_bold=bold,
        reading_order=index,
        bbox=BBox(
            x0=0,
            y0=index * 20,
            x1=500,
            y1=index * 20 + 15,
        ),
    )

def make_experience_block(
    block_id: str,
    text: str,
    reading_order: int,
    *,
    bold: bool = False,
    y: float = 0.0,
    section: str = "experience",
):
    return DocumentBlock(
        id=block_id,
        page=1,
        page_width=600.0,
        page_height=800.0,
        text=text,
        bbox={
            "x0": 40.0,
            "y0": y,
            "x1": 560.0,
            "y1": y + 20.0,
        },
        font_size=11.0,
        font_name="Arial",
        is_bold=bold,
        is_italic=False,
        reading_order=reading_order,
        column_id=0,
        section=section,
    )

def normalize_episode(episode):
    return {
        "role": episode.role,
        "company": episode.company,
        "location": episode.location,
        "start_date": episode.start_date,
        "end_date": episode.end_date,
        "is_current": episode.is_current,
        "description": episode.description,
        "bullet_points": list(episode.bullet_points),
        "technologies_mentioned": list(
            episode.technologies_mentioned
        ),
    }


def normalize_experiences(episodes):
    return [
        normalize_episode(episode)
        for episode in episodes
    ]

def test_phase18_shadow_single_block_parity():
    blocks = [
        make_experience_block(
            "b1",
            "Machine Learning Engineer @ OpenAI",
            1,
            bold=True,
        ),
        make_experience_block(
            "b2",
            "2024 - Present",
            2,
        ),
        make_experience_block(
            "b3",
            "Built production inference systems.",
            3,
        ),
    ]

    extractor = ExperienceExtractor()

    production = extractor.extract(blocks)
    shadow = extractor.extract_evidence_based(blocks)

    assert len(production) == 1
    assert len(shadow) == 1

    prod = normalize_episode(production[0])
    shad = normalize_episode(shadow[0])

    assert prod["role"] == shad["role"]
    assert prod["company"] == shad["company"]
    assert prod["start_date"] == shad["start_date"]
    assert prod["is_current"] == shad["is_current"]

def test_phase18_shadow_multiblock_parity():
    blocks = [
        make_experience_block(
            "b1",
            "Data Analyst Intern",
            1,
            bold=True,
        ),
        make_experience_block(
            "b2",
            "ComplyHappy Pvt Ltd",
            2,
        ),
        make_experience_block(
            "b3",
            "Jun 2023 - Aug 2023",
            3,
        ),
        make_experience_block(
            "b4",
            "Built automated reporting pipelines using Python.",
            4,
        ),
    ]

    extractor = ExperienceExtractor()

    production = extractor.extract(blocks)
    shadow = extractor.extract_evidence_based(blocks)

    assert len(production) == 1
    assert len(shadow) == 1

    prod = normalize_episode(production[0])
    shad = normalize_episode(shadow[0])

    assert prod["role"] == shad["role"]
    assert prod["company"] == shad["company"]
    assert prod["start_date"] == shad["start_date"]
    assert prod["end_date"] == shad["end_date"]

def test_phase18_shadow_missing_company_is_not_invented():
    blocks = [
        make_experience_block(
            "b1",
            "Data Analyst Intern",
            1,
            bold=True,
        ),
        make_experience_block(
            "b2",
            "Pune, India",
            2,
        ),
        make_experience_block(
            "b3",
            "Jun 2023 - Aug 2023",
            3,
        ),
    ]

    extractor = ExperienceExtractor()

    shadow = extractor.extract_evidence_based(blocks)

    assert len(shadow) == 1

    episode = shadow[0]

    assert episode.role == "Data Analyst Intern"
    assert episode.location == "Pune, India"

    # Final model currently uses a compatibility placeholder.
    assert episode.company == "Company Not Specified"

def test_phase18_shadow_multiple_experiences_do_not_merge():
    blocks = [
        make_experience_block(
            "b1",
            "Data Analyst",
            1,
            bold=True,
        ),
        make_experience_block(
            "b2",
            "Company A",
            2,
        ),
        make_experience_block(
            "b3",
            "2020 - 2022",
            3,
        ),
        make_experience_block(
            "b4",
            "Prepared dashboards.",
            4,
        ),
        make_experience_block(
            "b5",
            "ML Engineer",
            5,
            bold=True,
            y=220,
        ),
        make_experience_block(
            "b6",
            "Company B",
            6,
            y=245,
        ),
        make_experience_block(
            "b7",
            "2023 - Present",
            7,
            y=270,
        ),
        make_experience_block(
            "b8",
            "Built ML inference systems.",
            8,
            y=295,
        ),
    ]

    extractor = ExperienceExtractor()

    shadow = extractor.extract_evidence_based(blocks)

    assert len(shadow) == 2

    assert shadow[0].role == "Data Analyst"
    assert shadow[0].company == "Company A"

    assert shadow[1].role == "ML Engineer"
    assert shadow[1].company == "Company B"

def test_phase18_shadow_does_not_cross_project_section():
    blocks = [
        make_experience_block(
            "b1",
            "Software Engineer",
            1,
            bold=True,
        ),
        make_experience_block(
            "b2",
            "Acme Corp",
            2,
        ),
        make_experience_block(
            "b3",
            "2024 - Present",
            3,
        ),
        make_experience_block(
            "p1",
            "Projects",
            4,
            section="projects",
            bold=True,
        ),
        make_experience_block(
            "p2",
            "Built a recommendation system.",
            5,
            section="projects",
        ),
    ]

    extractor = ExperienceExtractor()

    shadow = extractor.extract_evidence_based(blocks)

    assert len(shadow) == 1

    episode = shadow[0]

    assert episode.role == "Software Engineer"
    assert episode.company == "Acme Corp"

    assert "recommendation system" not in (
        episode.description or ""
    ).lower()

def classify_phase18_parity(
    production,
    shadow,
):
    if not production and not shadow:
        return "MATCH"

    if len(production) == len(shadow):
        return "MATCH"

    if len(shadow) > len(production):
        return "IMPROVEMENT"

    if len(shadow) < len(production):
        return "REGRESSION"

    return "UNRESOLVED"

def test_phase18_parity_classification_is_deterministic():
    assert classify_phase18_parity([], []) == "MATCH"

    assert classify_phase18_parity(
        [{"role": "A"}],
        [{"role": "A"}],
    ) == "MATCH"

    assert classify_phase18_parity(
        [{"role": "A"}],
        [
            {"role": "A"},
            {"role": "B"},
        ],
    ) == "IMPROVEMENT"

    assert classify_phase18_parity(
        [
            {"role": "A"},
            {"role": "B"},
        ],
        [{"role": "A"}],
    ) == "REGRESSION"                    

def blocks(
    *texts,
    bold_indices=None,
):
    bold_indices = set(
        bold_indices or {0}
    )

    return [
        make_block(
            i,
            text,
            bold=i in bold_indices,
        )
        for i, text in enumerate(texts)
    ]


def test_complete_single_block_is_exact_match():
    result = ExperienceParityHarness().compare(
        "single_block",
        blocks(
            "Software Engineer @ OpenAI\n"
            "2024 - Present\n"
            "Built systems."
        ),
    )

    assert result.status == "match"
    assert result.legacy == result.new


def test_split_role_company_date_is_exact_match():
    result = ExperienceParityHarness().compare(
        "split_metadata",
        blocks(
            "Machine Learning Engineer",
            "OpenAI",
            "Jan 2024 - Present",
            "Built production inference systems.",
        ),
    )

    assert result.status == "match"
    assert len(result.new) == 1


def test_same_company_multiple_roles_is_explicit_improvement():
    result = ExperienceParityHarness().compare(
        "same_company_multiple_roles",
        blocks(
            "Software Engineer",
            "Google",
            "2022 - 2024",
            "Backend development.",
            "Data Scientist",
            "Google",
            "2024 - Present",
            "ML systems.",
            bold_indices={0},
        ),
    )

    assert result.status == "improvement"

    assert len(result.legacy) == 1
    assert len(result.new) == 2

    assert {
        item.role
        for item in result.new
    } == {
        "software engineer",
        "data scientist",
    }


def test_missing_date_does_not_create_regression():
    result = ExperienceParityHarness().compare(
        "missing_date",
        blocks(
            "Data Analyst Intern",
            "Acme Corp",
            "Analyzed dashboards and reports.",
        ),
    )

    assert result.status == "match"


def test_non_experience_section_is_not_included():
    result = ExperienceParityHarness().compare(
        "non_experience",
        [
            make_block(
                0,
                "Software Engineer\n"
                "Google\n"
                "2024 - Present",
                section="education",
            )
        ],
    )

    assert result.new == []


def test_report_classifies_cases_without_object_identity():
    harness = ExperienceParityHarness()

    report = harness.compare_cases(
        {
            "a": blocks(
                "Software Engineer @ OpenAI\n"
                "2024 - Present"
            ),
            "b": blocks(
                "Data Analyst Intern",
                "Acme Corp",
            ),
        }
    )

    assert report.total_cases == 2
    assert report.exact_matches == 2
    assert report.regressions == 0