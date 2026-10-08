from src.inference.segmentation_adapter import (
    ExperienceSegmentationAdapter,
)


# ============================================================
# HEADER / FORMAT VARIATIONS
# ============================================================


def test_role_company_location_date_single_block(
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

    assert result.metadata["candidate_count"] == 1
    assert result.metadata["segment_count"] == 1

    segment = result.segments[0]

    assert segment.role == "Data Analyst Intern"
    assert segment.company == "ComplyHappy Pvt Ltd"


def test_role_company_date_without_location(
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

    result = adapter.analyze([block])

    assert result.metadata["candidate_count"] == 1
    assert result.metadata["segment_count"] == 1

    segment = result.segments[0]

    assert segment.role == "Software Engineer"
    assert segment.company == "Google"


def test_role_at_company_format(
    make_document_block,
):
    block = make_document_block(
        text=(
            "Data Scientist @ Google\n"
            "2023 - 2024"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["candidate_count"] == 1
    assert result.metadata["segment_count"] == 1


def test_role_company_dash_format(
    make_document_block,
):
    block = make_document_block(
        text=(
            "Senior AI Engineer - TechVision AI Pvt. Ltd.\n"
            "Jan 2024 - Present"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["candidate_count"] == 1
    assert result.metadata["segment_count"] == 1


# ============================================================
# ANTI-MERGE
# ============================================================


def test_different_roles_and_companies_remain_separate(
    make_document_block,
):
    block_1 = make_document_block(
        text=(
            "Data Analyst\n"
            "ComplyHappy Pvt Ltd\n"
            "Pune, India\n"
            "19 Jun 2023 - 11 Aug 2023"
        ),
        section="experience",
    )

    block_2 = make_document_block(
        text=(
            "AI Engineer Intern\n"
            "Marva AI\n"
            "Pune, India\n"
            "06 Jan 2025 - 17 May 2025"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze(
        [block_1, block_2]
    )

    assert result.metadata["candidate_count"] == 2
    assert result.metadata["segment_count"] == 2

    assert result.segments[0].role == "Data Analyst"
    assert result.segments[0].company == "ComplyHappy Pvt Ltd"

    assert result.segments[1].role == "AI Engineer Intern"
    assert result.segments[1].company == "Marva AI"


def test_same_company_different_roles_remain_separate(
    make_document_block,
):
    block_1 = make_document_block(
        text=(
            "Software Engineer\n"
            "Google\n"
            "2023 - 2024"
        ),
        section="experience",
    )

    block_2 = make_document_block(
        text=(
            "Senior Software Engineer\n"
            "Google\n"
            "2024 - Present"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze(
        [block_1, block_2]
    )

    assert result.metadata["candidate_count"] == 2
    assert result.metadata["segment_count"] == 2


def test_same_role_different_companies_remain_separate(
    make_document_block,
):
    block_1 = make_document_block(
        text=(
            "Data Analyst\n"
            "Company A\n"
            "2022 - 2023"
        ),
        section="experience",
    )

    block_2 = make_document_block(
        text=(
            "Data Analyst\n"
            "Company B\n"
            "2023 - 2024"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze(
        [block_1, block_2]
    )

    assert result.metadata["candidate_count"] == 2
    assert result.metadata["segment_count"] == 2


# ============================================================
# SECTION CONTAMINATION
# ============================================================


def test_projects_section_is_not_experience(
    make_document_block,
):
    block = make_document_block(
        text=(
            "Software Engineer\n"
            "Project Alpha\n"
            "2024 - 2025"
        ),
        section="projects",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["anchor_count"] == 0
    assert result.metadata["candidate_count"] == 0
    assert result.metadata["segment_count"] == 0


def test_certifications_section_is_not_experience(
    make_document_block,
):
    block = make_document_block(
        text=(
            "AWS Solutions Architect\n"
            "Amazon Web Services\n"
            "2024"
        ),
        section="certifications",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["anchor_count"] == 0
    assert result.metadata["candidate_count"] == 0
    assert result.metadata["segment_count"] == 0


def test_education_section_is_not_experience(
    make_document_block,
):
    block = make_document_block(
        text=(
            "Software Engineering\n"
            "MIT\n"
            "2022 - 2026"
        ),
        section="education",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["anchor_count"] == 0
    assert result.metadata["candidate_count"] == 0
    assert result.metadata["segment_count"] == 0


# ============================================================
# WEAK / INCOMPLETE EVIDENCE
# ============================================================


def test_date_only_does_not_create_experience(
    make_document_block,
):
    block = make_document_block(
        text="2023 - 2024",
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["candidate_count"] == 0
    assert result.metadata["segment_count"] == 0


def test_generic_description_does_not_create_experience(
    make_document_block,
):
    block = make_document_block(
        text=(
            "Developed machine learning models "
            "for business prediction and automation."
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["candidate_count"] == 0
    assert result.metadata["segment_count"] == 0


# ============================================================
# CURRENT EXPERIENCE
# ============================================================


def test_current_experience_is_preserved(
    make_document_block,
):
    block = make_document_block(
        text=(
            "AI Engineer\n"
            "Marva AI\n"
            "2025 - Present"
        ),
        section="experience",
    )

    adapter = ExperienceSegmentationAdapter()

    result = adapter.analyze([block])

    assert result.metadata["candidate_count"] == 1
    assert result.metadata["segment_count"] == 1

    segment = result.segments[0]

    assert segment.role == "AI Engineer"
    assert segment.company == "Marva AI"