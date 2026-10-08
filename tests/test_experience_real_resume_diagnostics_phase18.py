from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline import ResumeAnalysisPipeline, normalize_unicode


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
}


@dataclass
class EpisodeDiagnostic:
    index: int
    role: Optional[str]
    company: Optional[str]
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    is_current: bool
    block_ids: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResumeDiagnostic:
    filename: str
    production: List[EpisodeDiagnostic]
    shadow: List[EpisodeDiagnostic]
    differences: List[str]


def _value(value: Any) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip()

    return value or None


def _episode_block_ids(episode: Any) -> List[str]:
    evidence = getattr(episode, "evidence", None)

    if not isinstance(evidence, dict):
        return []

    source_ids = evidence.get("source_block_ids")

    if isinstance(source_ids, list):
        return [
            str(item)
            for item in source_ids
            if item
        ]

    return []


def _episode_diagnostic(
    episode: Any,
    index: int,
) -> EpisodeDiagnostic:

    evidence = getattr(episode, "evidence", None)

    if not isinstance(evidence, dict):
        evidence = {}

    return EpisodeDiagnostic(
        index=index,
        role=_value(getattr(episode, "role", None)),
        company=_value(getattr(episode, "company", None)),
        location=_value(getattr(episode, "location", None)),
        start_date=_value(getattr(episode, "start_date", None)),
        end_date=_value(getattr(episode, "end_date", None)),
        is_current=bool(
            getattr(
                episode,
                "is_current",
                False,
            )
        ),
        block_ids=_episode_block_ids(episode),
        evidence=evidence,
    )


def _prepare_blocks(
    pipeline: ResumeAnalysisPipeline,
    file_path: Path,
):
    raw_blocks = pipeline.parser.parse(str(file_path))

    for block in raw_blocks:
        block.text = normalize_unicode(block.text)

    ordered_blocks = (
        pipeline.layout_engine
        .assign_columns_and_reading_order(
            raw_blocks
        )
    )

    cleaned_blocks = (
        pipeline.cleaner
        .filter_blocks(ordered_blocks)
    )

    return (
        pipeline.section_extractor
        .segment_document_by_sections(
            cleaned_blocks
        )
    )


def _run_production(
    pipeline: ResumeAnalysisPipeline,
    blocks,
):
    return pipeline.experience_extractor.extract(
        blocks
    )


def _run_shadow(
    pipeline: ResumeAnalysisPipeline,
    blocks,
):
    return (
        pipeline.experience_extractor
        .extract_evidence_based(blocks)
    )


def _field_differences(
    production: List[EpisodeDiagnostic],
    shadow: List[EpisodeDiagnostic],
) -> List[str]:

    differences: List[str] = []

    shared_count = min(
        len(production),
        len(shadow),
    )

    fields = (
        "role",
        "company",
        "location",
        "start_date",
        "end_date",
        "is_current",
    )

    for index in range(shared_count):
        prod = production[index]
        new = shadow[index]

        for field_name in fields:
            prod_value = getattr(
                prod,
                field_name,
            )

            shadow_value = getattr(
                new,
                field_name,
            )

            if prod_value != shadow_value:
                differences.append(
                    (
                        f"episode[{index}].{field_name}: "
                        f"production={prod_value!r}, "
                        f"shadow={shadow_value!r}"
                    )
                )

    if len(production) != len(shadow):
        differences.append(
            (
                "episode_count: "
                f"production={len(production)}, "
                f"shadow={len(shadow)}"
            )
        )

    return differences


def _print_episode(
    label: str,
    episode: EpisodeDiagnostic,
) -> None:

    print(
        f"  {label}[{episode.index}]"
    )

    print(
        f"    role      = {episode.role!r}"
    )

    print(
        f"    company   = {episode.company!r}"
    )

    print(
        f"    location  = {episode.location!r}"
    )

    print(
        f"    start     = {episode.start_date!r}"
    )

    print(
        f"    end       = {episode.end_date!r}"
    )

    print(
        f"    current   = {episode.is_current!r}"
    )

    print(
        f"    block_ids = {episode.block_ids!r}"
    )


def _print_block_context(
    blocks,
    interesting_ids: List[str],
) -> None:

    if not interesting_ids:
        return

    wanted = set(interesting_ids)

    print(
        "  BLOCK CONTEXT"
    )

    for block in blocks:
        if block.id not in wanted:
            continue

        print(
            f"    [{block.id}] "
            f"section={block.section!r} "
            f"page={block.page} "
            f"column={getattr(block, 'column_id', None)!r} "
            f"order={getattr(block, 'reading_order', None)!r}"
        )

        print(
            f"      text={block.text!r}"
        )

        print(
            f"      bold={block.is_bold!r} "
            f"heading={block.is_heading!r} "
            f"list_item={block.is_list_item!r}"
        )


def diagnose_resume(
    pipeline: ResumeAnalysisPipeline,
    file_path: Path,
) -> ResumeDiagnostic:

    blocks = _prepare_blocks(
        pipeline,
        file_path,
    )

    production = _run_production(
        pipeline,
        blocks,
    )

    shadow = _run_shadow(
        pipeline,
        blocks,
    )

    production_diagnostics = [
        _episode_diagnostic(
            episode,
            index,
        )
        for index, episode
        in enumerate(production)
    ]

    shadow_diagnostics = [
        _episode_diagnostic(
            episode,
            index,
        )
        for index, episode
        in enumerate(shadow)
    ]

    differences = _field_differences(
        production_diagnostics,
        shadow_diagnostics,
    )

    return ResumeDiagnostic(
        filename=file_path.name,
        production=production_diagnostics,
        shadow=shadow_diagnostics,
        differences=differences,
    )


def test_phase18b5_real_resume_diagnostics():

    root = Path("data/samples")

    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_SUFFIXES
    )

    assert files, (
        "No supported real-resume samples "
        "were found in data/samples."
    )

    pipeline = ResumeAnalysisPipeline()

    diagnostics: List[ResumeDiagnostic] = []

    for file_path in files:

        result = diagnose_resume(
            pipeline,
            file_path,
        )

        diagnostics.append(result)

        print()
        print("=" * 100)
        print(
            f"RESUME: {result.filename}"
        )
        print("=" * 100)

        print(
            f"PRODUCTION EPISODES: "
            f"{len(result.production)}"
        )

        for episode in result.production:
            _print_episode(
                "PRODUCTION",
                episode,
            )

        print()

        print(
            f"SHADOW EPISODES: "
            f"{len(result.shadow)}"
        )

        for episode in result.shadow:
            _print_episode(
                "SHADOW",
                episode,
            )

        if result.differences:

            print()
            print("DIFFERENCES:")

            for difference in result.differences:
                print(
                    f"  - {difference}"
                )

        else:

            print()
            print("DIFFERENCES: NONE")

    print()
    print("=" * 100)
    print("PHASE 18B.5 — DIAGNOSTIC SUMMARY")
    print("=" * 100)

    different = 0
    matching = 0

    for result in diagnostics:

        if result.differences:
            different += 1
        else:
            matching += 1

    print(
        f"RESUMES ANALYZED: {len(diagnostics)}"
    )

    print(
        f"NO FIELD DIFFERENCES: {matching}"
    )

    print(
        f"DIFFERENCES REQUIRING INVESTIGATION: "
        f"{different}"
    )

    # The purpose of this test is diagnosis.
    # It must fail only if the corpus cannot be executed.
    assert len(diagnostics) == len(files)