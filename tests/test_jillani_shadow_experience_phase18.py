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


def test_jillani_shadow_experience_extraction_forensics():
    """
    Phase 18B.5.5

    Diagnostic test for the evidence-based/shadow experience
    architecture.

    No production architecture is modified by this test.
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

    assert experience_blocks, (
        "Jillani regression: no experience blocks available "
        "for shadow extraction."
    )

    print()
    print("=" * 120)
    print("PHASE 18B.5.5 — JILLANI SHADOW EXPERIENCE FORENSICS")
    print("=" * 120)

    print(
        f"EXPERIENCE BLOCK COUNT: "
        f"{len(experience_blocks)}"
    )

    for index, block in enumerate(experience_blocks):

        print(
            f"[{index}] "
            f"id={block.id} | "
            f"order={getattr(block, 'reading_order', None)} | "
            f"text={block.text!r}"
        )

    candidates = (
        pipeline.experience_extractor
        .extract_evidence_based(tagged)
    )

    print()
    print("=" * 120)
    print("SHADOW EXPERIENCE EPISODES")
    print("=" * 120)

    print(f"SHADOW EPISODE COUNT: {len(candidates)}")

    for index, episode in enumerate(candidates):

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

    assert candidates, (
        "Jillani regression: evidence-based experience extraction "
        "returned zero episodes despite valid experience-section blocks."
    )