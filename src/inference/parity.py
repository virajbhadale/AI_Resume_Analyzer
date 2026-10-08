"""Semantic parity harness for legacy and next-generation experience inference.

The harness deliberately compares meaning, not implementation objects. It is
used as a migration gate while the evidence/candidate pipeline remains in
shadow mode.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.document_models import DocumentBlock, ExperienceEpisode
from src.inference.candidate_builder import MultiBlockCandidateBuilder
from src.inference.segmentation import ExperienceSegmenter as NewSegmenter
from src.inference.segmenter import ExperienceSegmenter as LegacySegmenter


@dataclass(frozen=True)
class ExperienceSnapshot:
    role: Optional[str]
    company: Optional[str]
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    is_current: bool


@dataclass
class ParityCaseResult:
    case_id: str
    legacy: List[ExperienceSnapshot] = field(default_factory=list)
    new: List[ExperienceSnapshot] = field(default_factory=list)
    status: str = "unclassified"
    mismatches: List[str] = field(default_factory=list)


@dataclass
class ParityReport:
    cases: List[ParityCaseResult] = field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.cases)

    @property
    def exact_matches(self) -> int:
        return sum(case.status == "match" for case in self.cases)

    @property
    def regressions(self) -> int:
        return sum(case.status == "regression" for case in self.cases)

    @property
    def improvements(self) -> int:
        return sum(case.status == "improvement" for case in self.cases)

    @property
    def semantic_differences(self) -> int:
        return sum(case.status == "difference" for case in self.cases)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "exact_matches": self.exact_matches,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "semantic_differences": self.semantic_differences,
            "cases": [
                {
                    "case_id": case.case_id,
                    "status": case.status,
                    "legacy": [
                        snapshot.__dict__
                        for snapshot in case.legacy
                    ],
                    "new": [
                        snapshot.__dict__
                        for snapshot in case.new
                    ],
                    "mismatches": list(case.mismatches),
                }
                for case in self.cases
            ],
        }


class ExperienceParityHarness:
    """Run both experience architectures against identical DocumentBlocks."""

    _FIELDS = (
        "role",
        "company",
        "location",
        "start_date",
        "end_date",
        "is_current",
    )

    def __init__(self) -> None:
        self.legacy_segmenter = LegacySegmenter()
        self.new_segmenter = NewSegmenter()
        self.new_builder = MultiBlockCandidateBuilder()

    @staticmethod
    def _normalize(value: Any) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip().casefold()

        if not text:
            return None

        return " ".join(text.split())

    @classmethod
    def _snapshot_legacy(
        cls,
        episode: ExperienceEpisode,
    ) -> ExperienceSnapshot:
        return ExperienceSnapshot(
            role=cls._normalize(episode.role),
            company=cls._normalize(episode.company),
            location=cls._normalize(episode.location),
            start_date=cls._normalize(episode.start_date),
            end_date=cls._normalize(episode.end_date),
            is_current=bool(episode.is_current),
        )

    @classmethod
    def _snapshot_new(
        cls,
        segment,
    ) -> ExperienceSnapshot:
        return ExperienceSnapshot(
            role=cls._normalize(segment.role),
            company=cls._normalize(segment.company),
            location=cls._normalize(segment.location),
            start_date=cls._normalize(segment.start_date),
            end_date=cls._normalize(segment.end_date),
            is_current=bool(segment.is_current),
        )

    @staticmethod
    def _sort_key(
        snapshot: ExperienceSnapshot,
    ) -> tuple:
        return (
            snapshot.start_date or "",
            snapshot.end_date or "",
            snapshot.company or "",
            snapshot.role or "",
        )

    @classmethod
    def _compare_fields(
        cls,
        legacy: Sequence[ExperienceSnapshot],
        new: Sequence[ExperienceSnapshot],
    ) -> List[str]:
        mismatches: List[str] = []

        for index, (left, right) in enumerate(
            zip(legacy, new)
        ):
            for field_name in cls._FIELDS:
                left_value = getattr(left, field_name)
                right_value = getattr(right, field_name)

                if left_value != right_value:
                    mismatches.append(
                        f"episode[{index}].{field_name}: "
                        f"legacy={left_value!r}, "
                        f"new={right_value!r}"
                    )

        if len(legacy) != len(new):
            mismatches.append(
                f"episode_count: "
                f"legacy={len(legacy)}, "
                f"new={len(new)}"
            )

        return mismatches

    @staticmethod
    def _is_improvement(
        legacy: Sequence[ExperienceSnapshot],
        new: Sequence[ExperienceSnapshot],
    ) -> bool:
        """Recognize narrow evidence of better boundary preservation."""

        if len(new) <= len(legacy):
            return False

        if len(new) != 2:
            return False

        if len(legacy) != 1:
            return False

        left, right = new

        if not left.role or not right.role:
            return False

        if left.role == right.role:
            return False

        if left.company != right.company:
            return False

        return (
            left.start_date != right.start_date
            or left.end_date != right.end_date
        )

    @staticmethod
    def _is_regression(
        legacy: Sequence[ExperienceSnapshot],
        new: Sequence[ExperienceSnapshot],
    ) -> bool:
        """Detect only high-confidence information loss."""

        if len(new) != len(legacy) or not legacy:
            return False

        for old, current in zip(legacy, new):
            same_identity = (
                old.role == current.role
                or old.company == current.company
                or (
                    old.start_date == current.start_date
                    and old.end_date == current.end_date
                )
            )

            if not same_identity:
                continue

            for field_name in ExperienceParityHarness._FIELDS:
                old_value = getattr(old, field_name)
                new_value = getattr(current, field_name)

                if old_value is not None and new_value is None:
                    return True

        return False

    def compare(
        self,
        case_id: str,
        blocks: Iterable[DocumentBlock],
    ) -> ParityCaseResult:

        blocks = [
            block
            for block in blocks
            if (
                not block.section
                or block.section.strip().lower() == "experience"
            )
        ]

        # The legacy segmenter currently emits diagnostic prints.
        # Keep those diagnostics out of parity output.
        sink = StringIO()

        with redirect_stdout(sink):
            legacy_episodes = (
                self.legacy_segmenter.segment_experiences(blocks)
            )

            candidates = (
                self.new_builder.build_from_blocks(blocks)
            )

            new_result = (
                self.new_segmenter.segment(candidates)
            )

        legacy = sorted(
            (
                self._snapshot_legacy(item)
                for item in legacy_episodes
            ),
            key=self._sort_key,
        )

        new = sorted(
            (
                self._snapshot_new(item)
                for item in new_result.segments
            ),
            key=self._sort_key,
        )

        mismatches = self._compare_fields(
            legacy,
            new,
        )

        if legacy == new:
            status = "match"

        elif self._is_improvement(
            legacy,
            new,
        ):
            status = "improvement"

        elif self._is_regression(
            legacy,
            new,
        ):
            status = "regression"

        else:
            status = "difference"

        return ParityCaseResult(
            case_id=case_id,
            legacy=legacy,
            new=new,
            status=status,
            mismatches=mismatches,
        )

    def compare_cases(
        self,
        cases: Dict[str, Iterable[DocumentBlock]],
    ) -> ParityReport:

        return ParityReport(
            cases=[
                self.compare(
                    case_id,
                    blocks,
                )
                for case_id, blocks in cases.items()
            ]
        )