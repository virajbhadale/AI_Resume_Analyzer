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

    tagged_blocks = (
        pipeline.section_extractor
        .segment_document_by_sections(cleaned_blocks)
    )

    return tagged_blocks


def _print_anchor_result(index, block, anchor):
    print()
    print("-" * 140)

    print(f"INDEX           : {index}")
    print(f"BLOCK ID        : {block.id}")
    print(f"PAGE            : {block.page}")
    print(
        f"READING ORDER   : "
        f"{getattr(block, 'reading_order', None)}"
    )
    print(f"SECTION         : {getattr(block, 'section', None)!r}")
    print(f"BOLD            : {getattr(block, 'is_bold', None)}")
    print(f"HEADING         : {getattr(block, 'is_heading', None)}")
    print(f"TEXT            : {block.text!r}")

    print()
    print("ANCHOR RESULT")
    print(f"  is_anchor_head : {getattr(anchor, 'is_anchor_head', None)!r}")
    print(f"  role           : {getattr(anchor, 'role', None)!r}")
    print(f"  company        : {getattr(anchor, 'company', None)!r}")
    print(f"  location       : {getattr(anchor, 'location', None)!r}")
    print(f"  start_date     : {getattr(anchor, 'start_date', None)!r}")
    print(f"  end_date       : {getattr(anchor, 'end_date', None)!r}")
    print(f"  is_current     : {getattr(anchor, 'is_current', None)!r}")
    print(f"  confidence     : {getattr(anchor, 'confidence', None)!r}")

    # Print any additional fields that the current AnchorAnalysis
    # implementation exposes. This is diagnostic only.
    if hasattr(anchor, "signals"):
        print(f"  signals        : {anchor.signals!r}")

    if hasattr(anchor, "evidence"):
        print(f"  evidence       : {anchor.evidence!r}")


def test_jillani_anchor_candidate_forensics():
    """
    Phase 18B.5.6

    Diagnostic forensic test.

    Purpose:
        Trace Jillani experience blocks through AnchorDetector and
        MultiBlockCandidateBuilder without modifying production code.

    This test intentionally does not assert expected anchor counts.
    The goal is to identify the first incorrect state.
    """

    assert JILLANI.exists(), (
        f"Missing real-resume fixture: {JILLANI}"
    )

    pipeline = ResumeAnalysisPipeline()

    tagged_blocks = _prepare_blocks(pipeline)

    experience_blocks = [
        block
        for block in tagged_blocks
        if block.section == "experience"
    ]

    assert len(tagged_blocks) == 167
    assert len(experience_blocks) == 69

    detector = pipeline.experience_extractor.segmenter.anchor_detector

    print()
    print("=" * 140)
    print("PHASE 18B.5.6 — JILLANI ANCHOR FORENSICS")
    print("=" * 140)

    print(f"TOTAL BLOCKS       : {len(tagged_blocks)}")
    print(f"EXPERIENCE BLOCKS  : {len(experience_blocks)}")

    anchor_results = []

    for index, block in enumerate(experience_blocks):

        anchor = detector.analyze_block(
            block.text,
            is_bold=bool(block.is_bold),
        )

        anchor_results.append(anchor)

        _print_anchor_result(
            index,
            block,
            anchor,
        )

    print()
    print("=" * 140)
    print("ANCHOR SUMMARY")
    print("=" * 140)

    anchor_heads = [
        (block, anchor)
        for block, anchor in zip(
            experience_blocks,
            anchor_results,
        )
        if getattr(anchor, "is_anchor_head", False)
    ]

    print(f"ANCHOR HEAD COUNT : {len(anchor_heads)}")

    for index, (block, anchor) in enumerate(anchor_heads):

        print(
            f"[{index}] "
            f"order={getattr(block, 'reading_order', None)} | "
            f"role={getattr(anchor, 'role', None)!r} | "
            f"company={getattr(anchor, 'company', None)!r} | "
            f"date={getattr(anchor, 'start_date', None)!r} | "
            f"text={block.text!r}"
        )

    print()
    print("=" * 140)
    print("EXPECTED STRUCTURAL EXPERIENCE HEADERS — DIAGNOSTIC ONLY")
    print("=" * 140)

    known_headers = (
        "PURELOGICS",
        "NVIDIA",
        "BLOCBELT",
        "GOOGLE-KAGGLE",
        "Crypto-Express",
        "Pakistan Freelancing Training Center",
    )

    for company in known_headers:

        matching_blocks = [
            block
            for block in experience_blocks
            if company.lower() in block.text.lower()
        ]

        print(
            f"{company!r}: "
            f"{len(matching_blocks)} matching block(s)"
        )

        for block in matching_blocks:
            print(
                f"    order={getattr(block, 'reading_order', None)} | "
                f"text={block.text!r}"
            )

    print()
    print("=" * 140)
    print("CANDIDATE BUILDER FORENSICS")
    print("=" * 140)

    builder = pipeline.experience_extractor.segmenter.candidate_builder

    candidates = builder.build_from_blocks(
        experience_blocks
    )

    print(f"CANDIDATE COUNT : {len(candidates)}")

    for index, candidate in enumerate(candidates):

        print()
        print(f"CANDIDATE {index}")
        print("-" * 140)

        print(f"candidate_id : {candidate.candidate_id}")
        print(f"block_ids    : {candidate.block_ids}")
        print(f"role        : {candidate.role!r}")
        print(f"company     : {candidate.company!r}")
        print(f"location    : {candidate.location!r}")
        print(f"start_date  : {candidate.start_date!r}")
        print(f"end_date    : {candidate.end_date!r}")
        print(f"is_current  : {candidate.is_current!r}")
        print(f"section     : {candidate.section!r}")
        print(f"description : {candidate.description!r}")
        print(f"confidence  : {candidate.confidence!r}")
        print(f"status      : {candidate.status!r}")

        if hasattr(candidate, "evidence"):
            print(f"evidence    : {candidate.evidence!r}")

    # Deterministic execution gates only.
    assert len(tagged_blocks) == 167
    assert len(experience_blocks) == 69