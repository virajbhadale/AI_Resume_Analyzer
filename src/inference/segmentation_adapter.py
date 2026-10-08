"""
Shadow adapter for the next-generation experience inference architecture.

Pipeline:

    DocumentBlock[]
        ↓
    AnchorDetector
        ↓
    AnchorCandidate[]
        ↓
    CandidateBuilder
        ↓
    ExperienceCandidate[]
        ↓
    ExperienceSegmenter
        ↓
    SegmentationResult

IMPORTANT
---------
This module is intentionally NOT connected to the production
ExperienceExtractor yet.

It is used during architectural migration and regression testing.
"""

from __future__ import annotations

from typing import Dict, List

from src.document_models import DocumentBlock

from src.inference.anchors import (
    AnchorCandidate,
    AnchorDetector,
)

from src.inference.candidates import (
    ExperienceCandidate,
)

from src.inference.candidate_builder import (
    CandidateBuilder,
)

from src.inference.segmentation import (
    ExperienceSegment,
    ExperienceSegmenter,
    SegmentationResult,
)


class ExperienceSegmentationAdapter:
    """
    Bridge between the existing DocumentBlock/Anchor architecture
    and the new candidate/segmentation architecture.

    This adapter does NOT create ExperienceEpisode objects.

    It exists specifically to support migration, diagnostics,
    regression testing, and side-by-side comparison with the
    legacy ExperienceSegmenter.
    """

    def __init__(
        self,
        anchor_detector: AnchorDetector | None = None,
        candidate_builder: CandidateBuilder | None = None,
        segmenter: ExperienceSegmenter | None = None,
    ) -> None:

        self.anchor_detector = (
            anchor_detector
            or AnchorDetector()
        )

        self.candidate_builder = (
            candidate_builder
            or CandidateBuilder()
        )

        self.segmenter = (
            segmenter
            or ExperienceSegmenter()
        )

    # ------------------------------------------------------------------
    # Block filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _eligible_blocks(
        blocks: List[DocumentBlock],
    ) -> List[DocumentBlock]:
        """
        Keep blocks that are eligible for experience inference.

        Canonical section metadata is respected.

        This prevents the shadow pipeline from accidentally treating
        skills, education, projects, certifications, etc. as work
        experience.
        """

        eligible: List[DocumentBlock] = []

        for block in blocks:

            if not block.text or not block.text.strip():
                continue

            section = (
                block.section.strip().lower()
                if block.section
                else None
            )

            if section:

                if section != "experience":
                    continue

            eligible.append(block)

        return eligible

    # ------------------------------------------------------------------
    # Anchor detection
    # ------------------------------------------------------------------

    def detect_anchors(
        self,
        blocks: List[DocumentBlock],
    ) -> List[AnchorCandidate]:
        """
        Detect anchors from eligible document blocks.

        Only strong anchor-head blocks are converted into candidates
        during this first migration stage.

        Multi-block ROLE → COMPANY → DATE assembly is intentionally
        deferred to the next migration step.
        """

        anchors: List[AnchorCandidate] = []

        for block in self._eligible_blocks(blocks):

            anchor = (
                self.anchor_detector.analyze_block(
                    block.text,
                    is_bold=block.is_bold,
                )
            )

            anchor.block_id = block.id

            if not anchor.is_anchor_head:
                continue

            anchors.append(anchor)

        return anchors

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def build_candidates(
        self,
        anchors: List[AnchorCandidate],
    ) -> List[ExperienceCandidate]:
        """
        Convert anchor observations into ExperienceCandidate objects.
        """

        return (
            self.candidate_builder
            .build_candidates(anchors)
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_candidates(
        self,
        candidates: List[ExperienceCandidate],
    ) -> SegmentationResult:
        """
        Group candidates into provisional experience segments.
        """

        return self.segmenter.segment(
            candidates
        )

    # ------------------------------------------------------------------
    # Full shadow pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        blocks: List[DocumentBlock],
    ) -> SegmentationResult:
        """
        Run the complete shadow pipeline.
        """

        anchors = self.detect_anchors(
            blocks
        )

        candidates = self.build_candidates(
            anchors
        )

        result = self.segment_candidates(
            candidates
        )

        # Preserve useful migration diagnostics.
        result.metadata.update(
            {
                "adapter": (
                    "ExperienceSegmentationAdapter"
                ),
                "anchor_count": len(anchors),
                "candidate_count": len(candidates),
                "segment_count": len(
                    result.segments
                ),
            }
        )

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def candidate_summary(
        candidates: List[ExperienceCandidate],
    ) -> List[Dict]:
        """
        Convert candidates into simple regression-friendly records.
        """

        return [
            {
                "candidate_id": candidate.candidate_id,
                "block_ids": list(
                    candidate.block_ids
                ),
                "role": candidate.role,
                "company": candidate.company,
                "location": candidate.location,
                "start_date": candidate.start_date,
                "end_date": candidate.end_date,
                "is_current": candidate.is_current,
                "confidence": candidate.confidence,
                "status": candidate.status.value,
            }
            for candidate in candidates
        ]

    @staticmethod
    def segment_summary(
        result: SegmentationResult,
    ) -> List[Dict]:
        """
        Convert provisional segments into regression-friendly
        diagnostic records.
        """

        summaries: List[Dict] = []

        for segment in result.segments:

            summaries.append(
                {
                    "segment_id": segment.segment_id,
                    "candidate_id": (
                        segment.candidate_id
                    ),
                    "block_ids": list(
                        segment.block_ids
                    ),
                    "anchor_block_ids": list(
                        segment.anchor_block_ids
                    ),
                    "block_count": (
                        segment.block_count
                    ),
                    "confidence": (
                        segment.confidence
                    ),
                    "status": (
                        segment.status.value
                    ),
                }
            )

        return summaries