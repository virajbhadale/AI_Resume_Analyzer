from src.pipeline import ResumeAnalysisPipeline


RESUME_PATH = "data/samples/Viraj_Resume.pdf"


def test_debug_experience_pipeline():
    print("\n")
    print("=" * 80)
    print("EXPERIENCE PIPELINE DEBUG")
    print("=" * 80)

    pipeline = ResumeAnalysisPipeline()

    # ---------------------------------------------------------
    # 1. PARSE
    # ---------------------------------------------------------
    raw_blocks = pipeline.parser.parse(RESUME_PATH)

    print("\n[1] RAW BLOCKS")
    print("Count:", len(raw_blocks))

    # ---------------------------------------------------------
    # 2. UNICODE NORMALIZATION
    # ---------------------------------------------------------
    from src.preprocessing.unicode_normalizer import normalize_unicode

    for block in raw_blocks:
        block.text = normalize_unicode(block.text)

    # ---------------------------------------------------------
    # 3. LAYOUT
    # ---------------------------------------------------------
    ordered_blocks = (
        pipeline.layout_engine
        .assign_columns_and_reading_order(raw_blocks)
    )

    print("\n[2] ORDERED BLOCKS")
    print("Count:", len(ordered_blocks))

    # ---------------------------------------------------------
    # 4. CLEANING
    # ---------------------------------------------------------
    cleaned_blocks = (
        pipeline.cleaner
        .filter_blocks(ordered_blocks)
    )

    print("\n[3] CLEANED BLOCKS")
    print("Count:", len(cleaned_blocks))

    # ---------------------------------------------------------
    # 5. SECTION TAGGING
    # ---------------------------------------------------------
    section_tagged_blocks = (
        pipeline.section_extractor
        .segment_document_by_sections(cleaned_blocks)
    )

    print("\n[4] SECTION TAGGED BLOCKS")
    print("Count:", len(section_tagged_blocks))

    # ---------------------------------------------------------
    # 6. SHOW EXPERIENCE BLOCKS
    # ---------------------------------------------------------
    experience_blocks = [
        block
        for block in section_tagged_blocks
        if block.section == "experience"
    ]

    print("\n[5] EXPERIENCE BLOCKS")
    print("Count:", len(experience_blocks))

    for i, block in enumerate(experience_blocks):
        print("\n--- EXPERIENCE BLOCK", i, "---")
        print("ID:", block.id)
        print("Section:", block.section)
        print("Bold:", block.is_bold)
        print("Heading:", block.is_heading)
        print("Text:")
        print(repr(block.text))

    # ---------------------------------------------------------
    # 7. EXPERIENCE EXTRACTOR
    # ---------------------------------------------------------
    experience_episodes = (
        pipeline.experience_extractor
        .extract(section_tagged_blocks)
    )

    print("\n[6] EXPERIENCE EPISODES")
    print("Count:", len(experience_episodes))

    for i, episode in enumerate(experience_episodes):
        print("\n--- EPISODE", i, "---")
        print("ID:", episode.id)
        print("Role:", episode.role)
        print("Company:", episode.company)
        print("Location:", episode.location)
        print("Start:", episode.start_date)
        print("End:", episode.end_date)
        print("Current:", episode.is_current)
        print("Description:", episode.description)
        print("Bullets:", episode.bullet_points)
        print("Technologies:", episode.technologies_mentioned)
        print("Confidence:", episode.confidence)

    # ---------------------------------------------------------
    # 8. RESUME BUILDER
    # ---------------------------------------------------------
    resume = pipeline.resume_builder.build(
        contact=pipeline.contact_extractor.extract(
            section_tagged_blocks
        ),
        current_title=None,
        domain_category=None,
        summary="",
        skills=pipeline.skill_extractor.extract(
            section_tagged_blocks
        ),
        experience=experience_episodes,
        education=pipeline.education_extractor.extract(
            section_tagged_blocks
        ),
        projects=pipeline.project_extractor.extract(
            section_tagged_blocks
        ),
        certifications=pipeline.cert_extractor.extract(
            section_tagged_blocks
        ),
        publications=[],
        awards=[],
        achievements=[],
        languages=[],
        interests=[],
        volunteer_experience=[],
        references=[],
        metadata={
            "source_file": RESUME_PATH,
            "total_blocks_processed": len(
                section_tagged_blocks
            ),
        },
    )

    # ---------------------------------------------------------
    # 9. BUILT RESUME
    # ---------------------------------------------------------
    print("\n[7] BUILT RESUME")

    print(
        "Resume type:",
        type(resume)
    )

    print(
        "Resume experiences:",
        getattr(resume, "experience", None)
    )

    print(
        "Resume experience count:",
        len(
            getattr(resume, "experience", [])
            or []
        )
    )

    # ---------------------------------------------------------
    # 10. MODEL DUMP
    # ---------------------------------------------------------
    dumped = resume.model_dump()

    print("\n[8] MODEL DUMP")

    print(
        "Dump keys:",
        dumped.keys()
    )

    print(
        "Dump experience:",
        dumped.get("experience")
    )

    print(
        "Dump experience count:",
        len(
            dumped.get("experience", [])
            or []
        )
    )

    print("\n")
    print("=" * 80)
    print("DEBUG COMPLETED")
    print("=" * 80)

    # This is intentionally not an assertion about correctness.
    assert True