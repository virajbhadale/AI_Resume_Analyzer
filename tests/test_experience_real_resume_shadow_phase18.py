"""
Phase 18B.4 — Real-Resume Shadow Comparison

Purpose
-------
Compare the current production experience extraction path against the
evidence-based shadow path on the real resume corpus.

IMPORTANT
---------
This test does NOT:
    - modify production extraction
    - replace ExperienceExtractor.extract()
    - migrate the shadow path
    - declare the shadow path superior
    - treat production output as ground truth

A mismatch is diagnostic evidence.

Classification policy:
    MATCH
        Production and shadow are semantically equivalent.

    UNRESOLVED
        A semantic difference exists but cannot be objectively classified
        without evidence investigation.

IMPROVEMENT / REGRESSION
    These require evidence-backed classification and are intentionally
    not inferred from "new != old".
    Phase 18B.5 is responsible for investigating unresolved differences.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

from src.inference.anchors import AnchorDetector
from src.inference.candidate_builder import MultiBlockCandidateBuilder
from src.inference.experience_pipeline import ExperienceEpisodeMaterializer
from src.inference.validation import ExperienceCandidateValidator
from src.pipeline import ResumeAnalysisPipeline, normalize_unicode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
BENCHMARK_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "experience_benchmark_manifest.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "Reports"
    / "PHASE_18B4_REAL_RESUME_SHADOW_COMPARISON.md"
)


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
}


COMPARISON_FIELDS = (
    "role",
    "company",
    "location",
    "start_date",
    "end_date",
    "is_current",
    "description",
    "bullet_points",
    "technologies_mentioned",
)


@dataclass(frozen=True)
class EpisodeSnapshot:
    role: Optional[str]
    company: Optional[str]
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    is_current: bool
    description: str
    bullet_points: List[str]
    technologies_mentioned: List[str]
    confidence: float
    provenance_present: bool


@dataclass(frozen=True)
class FieldDifference:
    episode_index: int
    field: str
    production: Any
    shadow: Any


@dataclass(frozen=True)
class ResumeComparison:
    resume: str
    production_episode_count: int
    shadow_episode_count: int
    classification: str
    differences: List[FieldDifference]
    production: List[EpisodeSnapshot]
    shadow: List[EpisodeSnapshot]

@dataclass(frozen=True)
class BenchmarkExperience:
    role: Optional[str]
    company: Optional[str]
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    is_current: Optional[bool]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    source: str
    status: str
    expected_experience_count: Optional[int]
    experiences: List[BenchmarkExperience]


@dataclass(frozen=True)
class BenchmarkComparison:
    case_id: str
    classification: str
    expected_count: Optional[int]
    actual_count: int
    matched: int
    missing: int
    unexpected: int
    details: List[str]

def _load_experience_benchmark_manifest() -> Dict[str, Any]:
    """
    Load the canonical experience benchmark.

    The benchmark is test data and is intentionally kept outside
    production extraction code.
    """
    assert BENCHMARK_MANIFEST_PATH.exists(), (
        "Missing experience benchmark manifest: "
        f"{BENCHMARK_MANIFEST_PATH}"
    )

    with BENCHMARK_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    assert manifest.get("version") == "1.0", (
        "Unsupported experience benchmark version: "
        f"{manifest.get('version')!r}"
    )

    assert manifest.get("benchmark") == "experience", (
        "Unexpected benchmark type: "
        f"{manifest.get('benchmark')!r}"
    )

    cases = manifest.get("cases")

    assert isinstance(cases, list), (
        "Experience benchmark 'cases' must be a list."
    )

    assert cases, (
        "Experience benchmark must contain at least one case."
    )

    return manifest

def _benchmark_cases(
    manifest: Dict[str, Any],
) -> List[BenchmarkCase]:
    """
    Convert raw manifest entries into deterministic benchmark cases.

    Cases are normative by default. A case explicitly marked
    'diagnostic' is excluded from canonical assertions until its
    semantics are resolved.
    """
    cases: List[BenchmarkCase] = []

    for raw_case in manifest["cases"]:
        status = raw_case.get("status", "normative")

        assert status in {"normative", "diagnostic"}, (
            f"Invalid benchmark status for "
            f"{raw_case.get('id')!r}: {status!r}"
        )

        raw_experiences = raw_case.get("experiences", [])

        assert isinstance(raw_experiences, list), (
            f"Invalid experiences list for "
            f"{raw_case.get('id')!r}"
        )

        experiences = [
            BenchmarkExperience(
                role=item.get("role"),
                company=item.get("company"),
                location=item.get("location"),
                start_date=item.get("start_date"),
                end_date=item.get("end_date"),
                is_current=item.get("is_current"),
            )
            for item in raw_experiences
        ]

        expected_count = raw_case.get(
            "expected_experience_count"
        )

        if status == "normative":
            assert expected_count is not None, (
                f"Normative benchmark case "
                f"{raw_case.get('id')!r} requires "
                f"'expected_experience_count'."
            )

            assert len(experiences) == expected_count, (
                f"Benchmark case {raw_case.get('id')!r} declares "
                f"{expected_count} experiences but provides "
                f"{len(experiences)}."
            )

        cases.append(
            BenchmarkCase(
                case_id=raw_case["id"],
                source=raw_case["source"],
                status=status,
                expected_experience_count=expected_count,
                experiences=experiences,
            )
        )

    return cases

def _benchmark_clean(value: Any) -> Any:
    """
    Normalize benchmark values without changing semantic meaning.
    """
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = " ".join(value.split()).strip()
        return cleaned.casefold() if cleaned else None

    return value

def _benchmark_episode_values(
    episode: Any,
) -> BenchmarkExperience:
    return BenchmarkExperience(
        role=getattr(episode, "role", None),
        company=getattr(episode, "company", None),
        location=getattr(episode, "location", None),
        start_date=getattr(episode, "start_date", None),
        end_date=getattr(episode, "end_date", None),
        is_current=getattr(episode, "is_current", None),
    )

def _benchmark_experience_match(
    expected: BenchmarkExperience,
    actual: BenchmarkExperience,
) -> bool:
    """
    Compare the semantic fields available in the benchmark.

    None in the expected benchmark means that the field is not
    asserted for that benchmark case.
    """
    fields = (
        "role",
        "company",
        "location",
        "start_date",
        "end_date",
        "is_current",
    )

    for field in fields:
        expected_value = getattr(expected, field)

        if expected_value is None:
            continue

        actual_value = getattr(actual, field)

        if _benchmark_clean(expected_value) != _benchmark_clean(
            actual_value
        ):
            return False

    return True

def _compare_against_benchmark(
    case: BenchmarkCase,
    shadow_episodes: Iterable[Any],
) -> BenchmarkComparison:
    """
    Compare shadow extraction against the canonical benchmark.

    Matching is one-to-one and order-independent. This prevents
    episode ordering from hiding a semantic match.
    """
    actual = [
        _benchmark_episode_values(episode)
        for episode in shadow_episodes
    ]

    if case.status == "diagnostic":
        return BenchmarkComparison(
            case_id=case.case_id,
            classification="UNRESOLVED",
            expected_count=case.expected_experience_count,
            actual_count=len(actual),
            matched=0,
            missing=0,
            unexpected=0,
            details=[
                "Diagnostic benchmark case; "
                "canonical semantic assertion is intentionally disabled."
            ],
        )

    expected = case.experiences

    unmatched_actual = set(range(len(actual)))
    matched = 0
    details: List[str] = []

    for expected_index, expected_episode in enumerate(expected):
        matching_index = None

        for actual_index in sorted(unmatched_actual):
            if _benchmark_experience_match(
                expected_episode,
                actual[actual_index],
            ):
                matching_index = actual_index
                break

        if matching_index is None:
            details.append(
                "Missing expected experience: "
                f"role={expected_episode.role!r}, "
                f"company={expected_episode.company!r}"
            )
            continue

        unmatched_actual.remove(matching_index)
        matched += 1

    missing = len(expected) - matched
    unexpected = len(unmatched_actual)

    if missing == 0 and unexpected == 0:
        classification = "MATCH"
    elif matched > 0 and missing == 0:
        classification = "IMPROVEMENT"
    elif matched == 0 and expected:
        classification = "REGRESSION"
    else:
        classification = "UNRESOLVED"

    for actual_index in sorted(unmatched_actual):
        episode = actual[actual_index]

        details.append(
            "Unexpected extracted experience: "
            f"role={episode.role!r}, "
            f"company={episode.company!r}"
        )

    return BenchmarkComparison(
        case_id=case.case_id,
        classification=classification,
        expected_count=case.expected_experience_count,
        actual_count=len(actual),
        matched=matched,
        missing=missing,
        unexpected=unexpected,
        details=details,
    )

def _benchmark_case_for_resume(
    case: BenchmarkCase,
) -> Path:
    source_path = PROJECT_ROOT / case.source

    assert source_path.exists(), (
        f"Benchmark source does not exist for "
        f"{case.case_id!r}: {source_path}"
    )

    return source_path

def test_canonical_experience_benchmark_manifest_is_valid():
    """
    Verify that the canonical benchmark can be loaded and validated.
    """
    manifest = _load_experience_benchmark_manifest()
    cases = _benchmark_cases(manifest)

    assert cases

    for case in cases:
        assert case.case_id
        assert case.source

        if case.status == "normative":
            assert case.expected_experience_count is not None
            assert (
                len(case.experiences)
                == case.expected_experience_count
            )


def test_shadow_matches_canonical_experience_benchmark():
    """
    Validate the evidence-based shadow extractor against the
    canonical semantic benchmark.

    Diagnostic cases are reported but intentionally do not fail
    the benchmark.
    """
    manifest = _load_experience_benchmark_manifest()
    cases = _benchmark_cases(manifest)

    pipeline = ResumeAnalysisPipeline()

    comparison_by_resume = {}

    for file_path in _resume_files():
        blocks = _prepare_blocks(
            pipeline,
            file_path,
        )

        shadow = _run_shadow(
            pipeline,
            blocks,
        )

        comparison_by_resume[file_path.name] = shadow

    failures: List[str] = []

    for case in cases:
        source_path = _benchmark_case_for_resume(case)

        shadow = comparison_by_resume.get(
            source_path.name
        )

        if shadow is None:
            failures.append(
                f"{case.case_id}: benchmark source was not "
                f"discovered by the real-resume corpus."
            )
            continue

        result = _compare_against_benchmark(
            case,
            shadow,
        )

        if case.status == "diagnostic":
            continue

        if result.classification != "MATCH":
            failures.append(
                f"{case.case_id}: "
                f"{result.classification}; "
                f"expected={result.expected_count}; "
                f"actual={result.actual_count}; "
                f"matched={result.matched}; "
                f"missing={result.missing}; "
                f"unexpected={result.unexpected}; "
                f"details={result.details}"
            )

    assert not failures, (
        "Canonical experience benchmark failures:\n"
        + "\n".join(failures)
    )        


def _clean(value: Any) -> Any:
    """
    Normalize comparison values without changing source semantics.
    """
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = " ".join(value.split()).strip()
        return cleaned.casefold() if cleaned else None

    if isinstance(value, list):
        return [
            _clean(item)
            for item in value
            if item is not None
        ]

    return value


def _episode_snapshot(episode: Any) -> EpisodeSnapshot:
    """
    Convert an ExperienceEpisode into a deterministic comparison snapshot.

    Provenance itself is not compared field-by-field in this phase.
    We only verify whether provenance exists on the shadow result.
    """
    evidence = getattr(episode, "evidence", None)

    return EpisodeSnapshot(
        role=getattr(episode, "role", None),
        company=getattr(episode, "company", None),
        location=getattr(episode, "location", None),
        start_date=getattr(episode, "start_date", None),
        end_date=getattr(episode, "end_date", None),
        is_current=bool(
            getattr(episode, "is_current", False)
        ),
        description=getattr(episode, "description", "") or "",
        bullet_points=list(
            getattr(episode, "bullet_points", []) or []
        ),
        technologies_mentioned=list(
            getattr(
                episode,
                "technologies_mentioned",
                [],
            )
            or []
        ),
        confidence=float(
            getattr(episode, "confidence", 0.0)
            or 0.0
        ),
        provenance_present=bool(evidence),
    )


def _compare_episode(
    production: EpisodeSnapshot,
    shadow: EpisodeSnapshot,
    episode_index: int,
) -> List[FieldDifference]:
    differences: List[FieldDifference] = []

    for field in COMPARISON_FIELDS:
        production_value = _clean(
            getattr(production, field)
        )
        shadow_value = _clean(
            getattr(shadow, field)
        )

        if production_value != shadow_value:
            differences.append(
                FieldDifference(
                    episode_index=episode_index,
                    field=field,
                    production=production_value,
                    shadow=shadow_value,
                )
            )

    return differences


def compare_resume(
    resume_name: str,
    production_episodes: Iterable[Any],
    shadow_episodes: Iterable[Any],
) -> ResumeComparison:
    """
    Compare production and shadow episodes.

    Episode matching is currently positional.

    This is deliberately conservative.

    We do NOT introduce a new fuzzy episode matcher in 18B.4 because
    doing so would hide boundary/ordering differences that are themselves
    important migration evidence.
    """
    production = [
        _episode_snapshot(episode)
        for episode in production_episodes
    ]

    shadow = [
        _episode_snapshot(episode)
        for episode in shadow_episodes
    ]

    differences: List[FieldDifference] = []

    common_count = min(
        len(production),
        len(shadow),
    )

    for index in range(common_count):
        differences.extend(
            _compare_episode(
                production[index],
                shadow[index],
                index,
            )
        )

    if len(production) != len(shadow):
        differences.append(
            FieldDifference(
                episode_index=-1,
                field="episode_count",
                production=len(production),
                shadow=len(shadow),
            )
        )

    classification = (
        "MATCH"
        if not differences
        else "UNRESOLVED"
    )

    return ResumeComparison(
        resume=resume_name,
        production_episode_count=len(production),
        shadow_episode_count=len(shadow),
        classification=classification,
        differences=differences,
        production=production,
        shadow=shadow,
    )


def _resume_files() -> List[Path]:
    if not SAMPLES_DIR.exists():
        return []

    return sorted(
        path
        for path in SAMPLES_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    )


def _prepare_blocks(
    pipeline: ResumeAnalysisPipeline,
    file_path: Path,
):
    """
    Reproduce the production preprocessing path exactly up to the
    experience extractor boundary.

    Both production and shadow extraction therefore receive identical
    DocumentBlocks.
    """
    raw_blocks = pipeline.parser.parse(
        str(file_path)
    )

    for block in raw_blocks:
        block.text = normalize_unicode(
            block.text
        )

    ordered_blocks = (
        pipeline.layout_engine
        .assign_columns_and_reading_order(
            raw_blocks
        )
    )

    cleaned_blocks = (
        pipeline.cleaner.filter_blocks(
            ordered_blocks
        )
    )

    section_tagged_blocks = (
        pipeline.section_extractor
        .segment_document_by_sections(
            cleaned_blocks
        )
    )

    return section_tagged_blocks


def _run_production(
    pipeline: ResumeAnalysisPipeline,
    blocks: list,
):
    """
    Current production path.

    This MUST remain ExperienceExtractor.extract().
    """
    return pipeline.experience_extractor.extract(
        blocks
    )


def _run_shadow(
    pipeline: ResumeAnalysisPipeline,
    blocks: list,
):
    """
    Current evidence-based shadow path.

    This MUST remain isolated from production extraction.
    """
    return (
        pipeline.experience_extractor
        .extract_evidence_based(blocks)
    )


def _serialize_comparison(
    comparison: ResumeComparison,
) -> Dict[str, Any]:
    return {
        "resume": comparison.resume,
        "production_episode_count": (
            comparison.production_episode_count
        ),
        "shadow_episode_count": (
            comparison.shadow_episode_count
        ),
        "classification": comparison.classification,
        "differences": [
            asdict(difference)
            for difference in comparison.differences
        ],
        "production": [
            asdict(snapshot)
            for snapshot in comparison.production
        ],
        "shadow": [
            asdict(snapshot)
            for snapshot in comparison.shadow
        ],
    }


def _build_markdown_report(
    comparisons: List[ResumeComparison],
) -> str:
    counts = {
        "MATCH": 0,
        "IMPROVEMENT": 0,
        "REGRESSION": 0,
        "UNRESOLVED": 0,
    }

    for comparison in comparisons:
        counts[comparison.classification] += 1

    lines: List[str] = []

    lines.append(
        "# Phase 18B.4 — Real-Resume Shadow Comparison"
    )
    lines.append("")
    lines.append(
        "## Status"
    )
    lines.append("")
    lines.append(
        "Shadow comparison executed without production migration."
    )
    lines.append("")
    lines.append(
        "Classification is conservative: a semantic difference "
        "is `UNRESOLVED` unless objective evidence supports a stronger "
        "classification."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Classification | Count |")
    lines.append("|---|---:|")
    lines.append(
        f"| MATCH | {counts['MATCH']} |"
    )
    lines.append(
        f"| IMPROVEMENT | {counts['IMPROVEMENT']} |"
    )
    lines.append(
        f"| REGRESSION | {counts['REGRESSION']} |"
    )
    lines.append(
        f"| UNRESOLVED | {counts['UNRESOLVED']} |"
    )
    lines.append("")

    lines.append("## Resume-Level Results")
    lines.append("")
    lines.append(
        "| Resume | Production | Shadow | Classification |"
    )
    lines.append(
        "|---|---:|---:|---|"
    )

    for comparison in comparisons:
        lines.append(
            "| "
            f"{comparison.resume} | "
            f"{comparison.production_episode_count} | "
            f"{comparison.shadow_episode_count} | "
            f"{comparison.classification} |"
        )

    lines.append("")

    for comparison in comparisons:
        if not comparison.differences:
            continue

        lines.append(
            f"## {comparison.resume}"
        )
        lines.append("")

        for difference in comparison.differences:
            lines.append(
                "- "
                f"episode={difference.episode_index}; "
                f"field=`{difference.field}`; "
                f"production={difference.production!r}; "
                f"shadow={difference.shadow!r}"
            )

        lines.append("")

    lines.append("## Migration Decision")
    lines.append("")
    lines.append(
        "**BLOCKED.** Phase 18B.4 does not authorize production "
        "migration. Differences require investigation before the "
        "migration gate."
    )
    lines.append("")

    lines.append("## Next Phase")
    lines.append("")
    lines.append(
        "Phase 18B.5 — Investigate meaningful differences using "
        "reproduce → trace → first incorrect state → root cause → "
        "generalized fix → regression."
    )
    lines.append("")

    return "\n".join(lines)


@pytest.fixture(scope="module")
def real_resume_comparisons() -> List[ResumeComparison]:
    """
    Execute the real-resume shadow comparison once for the test module.
    """
    files = _resume_files()

    if not files:
        pytest.fail(
            "No supported files found in data/samples."
        )

    pipeline = ResumeAnalysisPipeline()

    comparisons: List[ResumeComparison] = []

    for file_path in files:
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

        comparisons.append(
            compare_resume(
                file_path.name,
                production,
                shadow,
            )
        )

    return comparisons


def test_real_resume_corpus_is_discovered():
    files = _resume_files()

    assert files, (
        "The real-resume corpus was not discovered."
    )


def test_real_resume_shadow_executes_without_exception(
    real_resume_comparisons,
):
    """
    The primary execution gate for 18B.4.

    A semantic mismatch is allowed and recorded.
    Runtime failure is not.
    """
    assert real_resume_comparisons

    for comparison in real_resume_comparisons:
        assert comparison.resume
        assert comparison.production_episode_count >= 0
        assert comparison.shadow_episode_count >= 0


def test_shadow_produces_provenance_for_nonempty_episodes(
    real_resume_comparisons,
):
    """
    Every non-empty shadow episode should carry provenance.

    This verifies the Phase 18B.2/17 provenance contract survives
    the real-resume path.
    """
    for comparison in real_resume_comparisons:
        for episode in comparison.shadow:
            assert episode.provenance_present, (
                "Shadow episode lost provenance: "
                f"{comparison.resume}"
            )


def test_report_contains_all_resume_results(
    real_resume_comparisons,
):
    report = _build_markdown_report(
        real_resume_comparisons
    )

    for comparison in real_resume_comparisons:
        assert comparison.resume in report

    assert "MATCH" in report
    assert "UNRESOLVED" in report
    assert "Migration Decision" in report


def test_real_resume_corpus_has_expected_size():
    """
    The supplied project corpus historically contains 16 supported
    resume inputs.

    This is a corpus-integrity check, not a semantic ground-truth check.
    """
    files = _resume_files()

    assert len(files) == 16, (
        "Unexpected real-resume corpus size: "
        f"expected 16, found {len(files)}"
    )