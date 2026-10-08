from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SkillGapResult:
    matched_core: list[str] = field(default_factory=list)
    matched_secondary: list[str] = field(default_factory=list)

    missing_core: list[str] = field(default_factory=list)
    missing_secondary: list[str] = field(default_factory=list)

    total_matched: int = 0
    total_missing: int = 0
    coverage_percentage: float = 0.0

    # Extended skill-gap analysis
    core_coverage_percentage: float = 0.0
    secondary_coverage_percentage: float = 0.0

    required_skills: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)


def _clean_skills(skills: Iterable[str] | None) -> list[str]:
    """
    Normalize a skill collection while preserving order.
    """
    if not skills:
        return []

    result: list[str] = []
    seen: set[str] = set()

    for skill in skills:
        if not isinstance(skill, str):
            continue

        cleaned = skill.strip()

        if not cleaned:
            continue

        key = cleaned.casefold()

        if key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def _merge_unique(*skill_groups: Iterable[str]) -> list[str]:
    """
    Merge multiple skill collections while preserving order
    and removing case-insensitive duplicates.
    """
    result: list[str] = []
    seen: set[str] = set()

    for skills in skill_groups:
        for skill in skills:
            key = skill.casefold()

            if key not in seen:
                seen.add(key)
                result.append(skill)

    return result


def _coverage(matched_count: int, missing_count: int) -> float:
    """
    Calculate percentage coverage.
    """
    total = matched_count + missing_count

    if total == 0:
        return 0.0

    return round((matched_count / total) * 100, 2)


def calculate_skill_gap(
    matched_core: Iterable[str] | None = None,
    matched_secondary: Iterable[str] | None = None,
    missing_core: Iterable[str] | None = None,
    missing_secondary: Iterable[str] | None = None,
) -> SkillGapResult:
    """
    Build a structured skill-gap analysis from ATS skill results.

    The function preserves the original skill-gap fields while also
    providing aggregated matched/missing skills and separate coverage
    percentages for core and secondary skills.
    """

    matched_core_clean = _clean_skills(matched_core)
    matched_secondary_clean = _clean_skills(matched_secondary)

    missing_core_clean = _clean_skills(missing_core)
    missing_secondary_clean = _clean_skills(missing_secondary)

    # ---------------------------------------------------------
    # Basic totals
    # ---------------------------------------------------------

    total_matched = (
        len(matched_core_clean)
        + len(matched_secondary_clean)
    )

    total_missing = (
        len(missing_core_clean)
        + len(missing_secondary_clean)
    )

    total_skills = total_matched + total_missing

    coverage_percentage = (
        (total_matched / total_skills) * 100
        if total_skills
        else 0.0
    )

    # ---------------------------------------------------------
    # Separate coverage
    # ---------------------------------------------------------

    core_coverage_percentage = _coverage(
        len(matched_core_clean),
        len(missing_core_clean),
    )

    secondary_coverage_percentage = _coverage(
        len(matched_secondary_clean),
        len(missing_secondary_clean),
    )

    # ---------------------------------------------------------
    # Aggregated skill lists
    # ---------------------------------------------------------

    matched_skills = _merge_unique(
        matched_core_clean,
        matched_secondary_clean,
    )

    missing_skills = _merge_unique(
        missing_core_clean,
        missing_secondary_clean,
    )

    required_skills = _merge_unique(
        matched_skills,
        missing_skills,
    )

    return SkillGapResult(
        matched_core=matched_core_clean,
        matched_secondary=matched_secondary_clean,
        missing_core=missing_core_clean,
        missing_secondary=missing_secondary_clean,
        total_matched=total_matched,
        total_missing=total_missing,
        coverage_percentage=round(coverage_percentage, 2),
        core_coverage_percentage=core_coverage_percentage,
        secondary_coverage_percentage=secondary_coverage_percentage,
        required_skills=required_skills,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
    )