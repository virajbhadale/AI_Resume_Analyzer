from pathlib import Path

from src.pipeline import ResumeAnalysisPipeline
from src.cleaner import DocumentCleaner
from src.layout import LayoutEngine
from src.parser import DocumentParser
from src.section_extractor import SectionExtractor
from src.preprocessing.unicode_normalizer import normalize_unicode
from src.inference.anchors import AnchorDetector
from src.inference.candidate_builder import MultiBlockCandidateBuilder


JILLANI = Path("data/samples/Jillani Resume.pdf")


def prepare_experience_blocks():
    pipeline = ResumeAnalysisPipeline()

    raw_blocks = pipeline.parser.parse(str(JILLANI))

    for block in raw_blocks:
        block.text = normalize_unicode(block.text)

    ordered_blocks = pipeline.layout_engine.assign_columns_and_reading_order(
        raw_blocks
    )

    cleaned_blocks = pipeline.cleaner.filter_blocks(ordered_blocks)

    tagged_blocks = pipeline.section_extractor.segment_document_by_sections(
        cleaned_blocks
    )

    experience_blocks = [
        block
        for block in tagged_blocks
        if block.section == "experience"
        and not block.is_heading
    ]

    return experience_blocks


def test_jillani_direct_shadow_builder_forensics():
    blocks = prepare_experience_blocks()

    detector = AnchorDetector()
    builder = MultiBlockCandidateBuilder(
        anchor_detector=detector
    )

    print("\n" + "=" * 100)
    print("JILLANI — DIRECT MULTI-BLOCK SHADOW BUILDER FORENSICS")
    print("=" * 100)

    print(f"\nExperience blocks: {len(blocks)}")

    # ------------------------------------------------------------
    # STEP 1 — BLOCK EVIDENCE
    # ------------------------------------------------------------

    evidence = builder.build_block_evidence(blocks)

    print("\n" + "-" * 100)
    print("STEP 1 — BLOCK EVIDENCE")
    print("-" * 100)

    for index, item in enumerate(evidence):
        print(
            f"\n[{index:03d}] "
            f"order={getattr(item, 'reading_order', None)} "
            f"id={getattr(item, 'block_id', None)}"
        )

        print(f"TEXT: {getattr(item, 'text', '')!r}")

        print(
            "ROLE:",
            getattr(item, "role", None),
            "| COMPANY:",
            getattr(item, "company", None),
        )

        print(
            "START:",
            getattr(item, "start_date", None),
            "| END:",
            getattr(item, "end_date", None),
        )

        print(
            "LOCATION:",
            getattr(item, "location", None),
        )

        metadata = getattr(item, "metadata", {}) or {}

        print(
            "is_anchor_head:",
            metadata.get("is_anchor_head"),
        )

        print(
            "section:",
            getattr(item, "section", None),
        )

    # ------------------------------------------------------------
    # STEP 2 — CANDIDATE HEAD DECISIONS
    # ------------------------------------------------------------

    print("\n" + "-" * 100)
    print("STEP 2 — CANDIDATE HEAD DECISIONS")
    print("-" * 100)

    head_count = 0

    for index, item in enumerate(evidence):

        try:
            is_head = builder._is_candidate_head(
                item,
                evidence,
                index,
            )
        except TypeError:
            # Compatibility with implementations whose helper
            # accepts a different signature.
            try:
                is_head = builder._is_candidate_head(
                    item,
                    index,
                    evidence,
                )
            except TypeError:
                is_head = "UNSUPPORTED_SIGNATURE"

        if is_head is True:
            head_count += 1

        print(
            f"[{index:03d}] "
            f"{getattr(item, 'text', '')!r} "
            f"→ candidate_head={is_head}"
        )

    print(f"\nCANDIDATE HEAD COUNT: {head_count}")

    # ------------------------------------------------------------
    # STEP 3 — ACTUAL MULTI-BLOCK CANDIDATES
    # ------------------------------------------------------------

    print("\n" + "-" * 100)
    print("STEP 3 — ACTUAL MULTI-BLOCK CANDIDATES")
    print("-" * 100)

    candidates = builder.build_from_blocks(blocks)

    print(f"\nCANDIDATE COUNT: {len(candidates)}")

    for index, candidate in enumerate(candidates):

        print("\n" + "=" * 80)
        print(f"CANDIDATE {index + 1}")
        print("=" * 80)

        print("candidate_id:", candidate.candidate_id)
        print("block_ids:", candidate.block_ids)

        print("role:", candidate.role)
        print("company:", candidate.company)
        print("location:", candidate.location)

        print("start_date:", candidate.start_date)
        print("end_date:", candidate.end_date)
        print("is_current:", candidate.is_current)

        print(
            "description_block_ids:",
            candidate.description_block_ids,
        )

        print("description:", candidate.description)
        print("confidence:", candidate.confidence)
        print("status:", candidate.status)

        print("metadata:")

        for key, value in (candidate.metadata or {}).items():
            print(f"  {key}: {value!r}")

        print("evidence:")

        for signal in candidate.evidence:
            print(
                "  -",
                getattr(signal, "signal_type", None),
                "| value=",
                getattr(signal, "value", None),
                "| confidence=",
                getattr(signal, "confidence", None),
                "| polarity=",
                getattr(signal, "polarity", None),
                "| source_block=",
                getattr(signal, "source_block_id", None),
            )

    # ------------------------------------------------------------
    # STEP 4 — STRUCTURAL ASSERTION ONLY
    # ------------------------------------------------------------

    # This is intentionally NOT asserting expected Jillani
    # experience counts yet.
    #
    # The purpose of this test is forensic diagnosis.
    assert len(blocks) > 0
    assert evidence