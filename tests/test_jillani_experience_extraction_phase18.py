from __future__ import annotations

from pathlib import Path

from src.pipeline import ResumeAnalysisPipeline, normalize_unicode


JILLANI = Path("data/samples/Jillani Resume.pdf")


def _prepare_blocks(pipeline: ResumeAnalysisPipeline):
    raw_blocks = pipeline.parser.parse(str(JILLANI))

    for block in raw_blocks:
        block.text = normalize_unicode(block.text)

    ordered_blocks = (
        pipeline.layout_engine
        .assign_columns_and_reading_order(raw_blocks)
    )

    cleaned_blocks = (
        pipeline.cleaner
        .filter_blocks(ordered_blocks)
    )

    return cleaned_blocks


def test_jillani_production_experience_extraction_forensics():
    """
    Phase 18B.5.5

    Diagnostic regression test for Jillani production experience
    extraction after the section-detection fix.

    This test intentionally does not prescribe the final expected
    number of experiences yet. Its purpose is to establish the
    actual production extraction state before modifying segmentation.
    """

    assert JILLANI.exists(), (
        f"Missing real-resume fixture: {JILLANI}"
    )

    pipeline = ResumeAnalysisPipeline()

    blocks = _prepare_blocks(pipeline)

    tagged = (
        pipeline.section_extractor
        .segment_document_by_sections(blocks)
    )

    experience_blocks = [
        block
        for block in tagged
        if block.section == "experience"
    ]

    print()
    print("=" * 120)
    print("PHASE 18B.5.5 — JILLANI PRODUCTION EXPERIENCE FORENSICS")
    print("=" * 120)

    print(f"TOTAL BLOCKS           : {len(tagged)}")
    print(
        f"EXPERIENCE BLOCK COUNT : "
        f"{len(experience_blocks)}"
    )

    print()
    print("EXPERIENCE BLOCKS")
    print("-" * 120)

    for index, block in enumerate(experience_blocks):

        print(
            f"[{index}] "
            f"id={block.id} | "
            f"page={block.page} | "
            f"order={getattr(block, 'reading_order', None)} | "
            f"text={block.text!r}"
        )

    assert experience_blocks, (
        "Jillani regression: no experience blocks reached "
        "ExperienceExtractor."
    )

    episodes = pipeline.experience_extractor.extract(tagged)

    print()
    print("=" * 120)
    print("PRODUCTION EXPERIENCE EPISODES")
    print("=" * 120)

    print(f"EPISODE COUNT: {len(episodes)}")

    for index, episode in enumerate(episodes):

        print()
        print(f"EPISODE {index}")
        print("-" * 120)

        print(f"ID          : {episode.id}")
        print(f"ROLE        : {episode.role!r}")
        print(f"COMPANY     : {episode.company!r}")
        print(f"LOCATION    : {episode.location!r}")
        print(f"START DATE  : {episode.start_date!r}")
        print(f"END DATE    : {episode.end_date!r}")
        print(f"IS CURRENT  : {episode.is_current!r}")
        print(f"DESCRIPTION : {episode.description!r}")
        print(
            f"BULLETS     : "
            f"{episode.bullet_points!r}"
        )
        print(
            f"TECHNOLOGIES: "
            f"{episode.technologies_mentioned!r}"
        )
        print(f"CONFIDENCE  : {episode.confidence!r}")

    # The section-detection regression must remain protected.
    assert len(tagged) == 167

    # The key forensic gate:
    # production extraction must execute without losing all
    # experience information after the section fix.
    assert episodes, (
        "Jillani regression: production ExperienceExtractor "
        "returned zero experience episodes despite valid "
        "experience-section blocks."
    )