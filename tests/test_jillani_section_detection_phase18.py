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


def _print_block(block, index: int) -> None:
    print(
        "\n"
        + "-" * 120
    )

    print(f"INDEX          : {index}")
    print(f"BLOCK ID       : {block.id}")
    print(f"PAGE           : {block.page}")
    print(f"COLUMN         : {getattr(block, 'column_id', None)}")
    print(
        f"READING ORDER  : "
        f"{getattr(block, 'reading_order', None)}"
    )

    print(f"BBOX           : {block.bbox}")

    print(
        f"FONT SIZE      : "
        f"{getattr(block, 'font_size', None)}"
    )

    print(
        f"BOLD           : "
        f"{getattr(block, 'is_bold', None)}"
    )

    print(
        f"HEADING        : "
        f"{getattr(block, 'is_heading', None)}"
    )

    print(
        f"LIST ITEM      : "
        f"{getattr(block, 'is_list_item', None)}"
    )

    print(
        f"SEMANTIC LABEL : "
        f"{getattr(block, 'semantic_label', None)}"
    )

    print(
        f"TEXT           : "
        f"{block.text!r}"
    )


def test_jillani_section_assignment_forensics():

    assert JILLANI.exists(), (
        f"Missing real-resume fixture: {JILLANI}"
    )

    pipeline = ResumeAnalysisPipeline()

    blocks = _prepare_blocks(pipeline)

    tagged = (
        pipeline.section_extractor
        .segment_document_by_sections(blocks)
    )

    assert len(tagged) == len(blocks)

    section_counts = {}

    for block in tagged:
        section = block.section or "UNASSIGNED"

        section_counts[section] = (
            section_counts.get(section, 0) + 1
        )

    print()
    print("=" * 120)
    print("PHASE 18B.5.2 — JILLANI SECTION FORENSICS")
    print("=" * 120)

    print(
        f"TOTAL BLOCKS: {len(tagged)}"
    )

    print(
        f"SECTION COUNTS: {section_counts}"
    )

    print()
    print("=" * 120)
    print("ALL BLOCK ASSIGNMENTS")
    print("=" * 120)

    for index, block in enumerate(tagged):

        _print_block(
            block,
            index,
        )

        print(
            f"SECTION        : "
            f"{block.section!r}"
        )

    print()
    print("=" * 120)
    print("HEADING CLASSIFICATION FORENSICS")
    print("=" * 120)

    for index, block in enumerate(tagged):

        if not block.text.strip():
            continue

        result = (
            pipeline.section_extractor
            .classify_heading(
                block.text,
                is_bold=bool(block.is_bold),
                font_size=(
                    getattr(
                        block,
                        "font_size",
                        10.0,
                    )
                    or 10.0
                ),
            )
        )

        if result is not None:

            print(
                f"[{index}] "
                f"text={block.text!r} "
                f"section={block.section!r} "
                f"classification={result!r}"
            )

    print()
    print("=" * 120)
    print("EXPERIENCE-CANDIDATE BLOCKS")
    print("=" * 120)

    experience_blocks = [
        block
        for block in tagged
        if block.section == "experience"
    ]

    print(
        f"EXPERIENCE BLOCK COUNT: "
        f"{len(experience_blocks)}"
    )

    for block in experience_blocks:
        print(
            f"{block.id} | "
            f"page={block.page} | "
            f"order={getattr(block, 'reading_order', None)} | "
            f"text={block.text!r}"
        )

    # This is diagnostic, not a correctness assertion.
    #
    # The current defect is precisely that Jillani reaches
    # section extraction without an experience section.
    #
    # Therefore we only require deterministic execution.
    assert len(tagged) == 167

    experience_blocks = [
        block
        for block in tagged
        if block.section == "experience"
    ]

    experience_headings = [
        block
        for block in tagged
        if (
            block.section == "experience"
            and block.is_heading
        )
    ]

    assert experience_headings, (
        "Jillani regression: no Experience section heading was detected."
    )

    assert experience_blocks, (
        "Jillani regression: decorative Experience heading was not "
        "recognized, so no blocks were assigned to the experience section."
    )