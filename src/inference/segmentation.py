from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.inference.evidence_models import (
    AnchorEvidence,
    AnchorType,
    BoundaryEvidence,
    BoundaryType,
)
from src.inference.candidates import ExperienceCandidate


# ============================================================================
# Segment status
# ============================================================================


class SegmentStatus(str, Enum):
    """
    Lifecycle state of a provisional experience segment.

    Segmentation does not create a final ExperienceEpisode.
    """

    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    REJECTED = "rejected"


# ============================================================================
# Experience segment
# ============================================================================


class ExperienceSegment(BaseModel):
    """
    Provisional group of document blocks representing one possible
    professional experience.

    This is intentionally NOT an ExperienceEpisode.

    Segmentation answers:

        "Which blocks probably belong together?"

    Later stages answer:

        "Is this actually a valid experience?"
    """

    segment_id: str

    block_ids: List[str] = Field(
        default_factory=list
    )

    candidate_id: Optional[str] = None

    anchor_block_ids: List[str] = Field(
        default_factory=list
    )

    
    role: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False

    boundary_before: Optional[BoundaryEvidence] = None

    boundary_after: Optional[BoundaryEvidence] = None

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    status: SegmentStatus = (
        SegmentStatus.PROVISIONAL
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


    def add_block(
        self,
        block_id: str,
    ) -> None:
        """Add a block without duplication."""

        if block_id not in self.block_ids:
            self.block_ids.append(block_id)

    def add_anchor_block(
        self,
        block_id: str,
    ) -> None:
        """Register a block contributing anchor evidence."""

        if block_id not in self.anchor_block_ids:
            self.anchor_block_ids.append(block_id)

    @property
    def has_anchor(self) -> bool:
        return bool(self.anchor_block_ids)

    @property
    def block_count(self) -> int:
        return len(self.block_ids)


# ============================================================================
# Segmentation result
# ============================================================================


class SegmentationResult(BaseModel):
    """
    Result returned by the segmenter.

    Keeps segments and boundary evidence together so downstream
    validation/confidence stages can inspect the reasoning.
    """

    segments: List[ExperienceSegment] = Field(
        default_factory=list
    )

    boundaries: List[BoundaryEvidence] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================================
# Experience segmenter
# ============================================================================


class ExperienceSegmenter:
    """
    Groups provisional experience candidates into segments.

    Important:
        This class does NOT validate or finalize experiences.

    It uses existing candidate/anchor evidence and preserves
    separation between:

        detection
        segmentation
        validation
        resolution
    """

    def __init__(
        self,
        minimum_anchor_score: float = 0.50,
    ):
        self.minimum_anchor_score = minimum_anchor_score

    # ------------------------------------------------------------------------
    # Anchor evaluation
    # ------------------------------------------------------------------------

    def is_anchor(
        self,
        anchor: Optional[AnchorEvidence],
    ) -> bool:
        """
        Determine whether anchor evidence is strong enough to start
        a provisional experience segment.
        """

        if anchor is None:
            return False

        if anchor.score < self.minimum_anchor_score:
            return False

        return anchor.anchor_type != AnchorType.UNKNOWN

    # ------------------------------------------------------------------------
    # Boundary creation
    # ------------------------------------------------------------------------

    def detect_boundary(
        self,
        previous: ExperienceCandidate,
        current: ExperienceCandidate,
    ) -> BoundaryEvidence:
        """
        Produce structured boundary evidence between two candidates.

        Important
        ---------
        This method generates evidence. It does not create an
        ExperienceEpisode and does not perform final validation.

        Multiple independent signals are preserved so downstream
        segmentation can reason about both supporting and contradictory
        evidence.
        """

        boundary = BoundaryEvidence()

        # ------------------------------------------------------------------
        # Source block provenance
        # ------------------------------------------------------------------

        for block_id in previous.block_ids:
            boundary.add_previous_block(block_id)

        for block_id in current.block_ids:
            boundary.add_next_block(block_id)

        supporting = []
        contradictory = []

        # ------------------------------------------------------------------
        # ROLE EVIDENCE
        # ------------------------------------------------------------------

        role_changed = False
        role_same = False

        if previous.role and current.role:
            previous_role = previous.role.strip().lower()
            current_role = current.role.strip().lower()

            if previous_role != current_role:
                role_changed = True

                supporting.append(
                    self._make_boundary_signal(
                        signal_type="role_change",
                        value={
                            "previous": previous.role,
                            "current": current.role,
                        },
                        confidence=0.95,
                        explanation=(
                            "Previous and current candidates contain "
                            "different role evidence."
                        ),
                    )
                )
            else:
                role_same = True

                contradictory.append(
                    self._make_boundary_signal(
                        signal_type="same_role",
                        value=previous.role,
                        confidence=0.90,
                        explanation=(
                            "Previous and current candidates contain "
                            "the same role evidence."
                        ),
                    )
                )

        # ------------------------------------------------------------------
        # COMPANY EVIDENCE
        # ------------------------------------------------------------------

        company_changed = False
        company_same = False

        if previous.company and current.company:
            previous_company = previous.company.strip().lower()
            current_company = current.company.strip().lower()

            if previous_company != current_company:
                company_changed = True

                supporting.append(
                    self._make_boundary_signal(
                        signal_type="company_change",
                        value={
                            "previous": previous.company,
                            "current": current.company,
                        },
                        confidence=0.90,
                        explanation=(
                            "Previous and current candidates contain "
                            "different company evidence."
                        ),
                    )
                )
            else:
                company_same = True

                contradictory.append(
                    self._make_boundary_signal(
                        signal_type="same_company",
                        value=previous.company,
                        confidence=0.90,
                        explanation=(
                            "Previous and current candidates contain "
                            "the same company evidence."
                        ),
                    )
                )

        # ------------------------------------------------------------------
        # DATE EVIDENCE
        # ------------------------------------------------------------------

        dates_changed = self._has_distinct_date_range(
            previous,
            current,
        )

        if dates_changed:
            supporting.append(
                self._make_boundary_signal(
                    signal_type="date_range_change",
                    value={
                        "previous": (
                            previous.start_date,
                            previous.end_date,
                            previous.is_current,
                        ),
                        "current": (
                            current.start_date,
                            current.end_date,
                            current.is_current,
                        ),
                    },
                    confidence=0.88,
                    explanation=(
                        "Previous and current candidates contain "
                        "distinct date ranges."
                    ),
                )
            )

        # ------------------------------------------------------------------
        # CONTINUATION EVIDENCE
        # ------------------------------------------------------------------
        #
        # Same role/company with no date conflict is strong evidence that
        # content may belong to the same experience.
        #
        # This is especially important for:
        #
        #     Senior Machine Learning
        #     Engineer
        #
        #     ABC Technologies
        #
        # where lexical fragments may not independently represent roles.
        # ------------------------------------------------------------------

        if role_same and (
            company_same
            or (
                not previous.company
                and not current.company
            )
        ):
            contradictory.append(
                self._make_boundary_signal(
                    signal_type="experience_continuation",
                    value={
                        "role": previous.role,
                        "company": previous.company,
                    },
                    confidence=0.92,
                    explanation=(
                        "Matching role/company evidence supports "
                        "continuation of the same experience."
                    ),
                )
            )

        # ------------------------------------------------------------------
        # Determine boundary classification.
        #
        # Classification happens AFTER evidence collection so that
        # supporting and contradictory evidence are both preserved.
        # ------------------------------------------------------------------

        if role_changed:
            boundary.boundary_type = BoundaryType.NEW_ROLE

            # A role change is strong even when the company remains the same.
            score = 0.95

            if dates_changed:
                score = max(score, 0.97)

            if company_changed:
                score = max(score, 0.99)

            boundary.score = score

            boundary.explanation = (
                "Different role evidence indicates a possible new "
                "professional experience."
            )

        elif company_changed:
            boundary.boundary_type = BoundaryType.NEW_COMPANY

            score = 0.90

            if dates_changed:
                score = max(score, 0.94)

            boundary.score = score

            boundary.explanation = (
                "Different company evidence indicates a possible "
                "new professional experience."
            )

        elif dates_changed:
            boundary.boundary_type = BoundaryType.NEW_DATE_RANGE
            boundary.score = 0.88

            boundary.explanation = (
                "Distinct date ranges provide evidence of a possible "
                "experience boundary."
            )

        elif role_same or company_same:
            boundary.boundary_type = BoundaryType.DESCRIPTION_CONTINUATION
            boundary.score = 0.15

            boundary.explanation = (
                "Matching role/company evidence supports continuation "
                "rather than a new experience boundary."
            )

        else:
            boundary.boundary_type = BoundaryType.SAME_EXPERIENCE
            boundary.score = 0.20

            boundary.explanation = (
                "No strong independent boundary evidence was detected."
            )

        # ------------------------------------------------------------------
        # Preserve all evidence.
        # ------------------------------------------------------------------

        for signal in supporting:
            boundary.add_supporting_evidence(signal)

        for signal in contradictory:
            boundary.add_contradictory_evidence(signal)

        return boundary

    @staticmethod
    def _make_boundary_signal(
        signal_type: str,
        value: Any,
        confidence: float,
        explanation: str,
    ):
        """
        Create a lightweight EvidenceSignal for boundary reasoning.

        BoundaryEvidence stores the actual signal objects so downstream
        stages can inspect the evidence that produced the boundary.
        """

        from src.inference.evidence_models import (
            EvidenceSignal,
            EvidenceType,
        )

        signal_type_map = {
            "role_change": EvidenceType.ROLE,
            "same_role": EvidenceType.ROLE,
            "company_change": EvidenceType.COMPANY,
            "same_company": EvidenceType.COMPANY,
            "date_range_change": EvidenceType.DATE,
            "experience_continuation": EvidenceType.STRUCTURAL,
        }

        return EvidenceSignal(
            signal_type=signal_type_map.get(
                signal_type,
                EvidenceType.STRUCTURAL,
            ),
            value=value,
            confidence=confidence,
            explanation=explanation,
            metadata={
                "boundary_signal": signal_type,
            },
        )

    # ------------------------------------------------------------------------
    # Date comparison
    # ------------------------------------------------------------------------

    @staticmethod
    def _has_distinct_date_range(
        previous: ExperienceCandidate,
        current: ExperienceCandidate,
    ) -> bool:
        """
        Detect whether both candidates have date information that
        differs.

        We deliberately avoid complex chronological reasoning here.
        Metadata validation will handle date consistency later.
        """

        previous_dates = (
            previous.start_date,
            previous.end_date,
            previous.is_current,
        )

        current_dates = (
            current.start_date,
            current.end_date,
            current.is_current,
        )

        previous_has_date = any(
            value is not None
            for value in previous_dates
        )

        current_has_date = any(
            value is not None
            for value in current_dates
        )

        if not previous_has_date or not current_has_date:
            return False

        return previous_dates != current_dates

    # ------------------------------------------------------------------------
    # Candidate grouping
    # ------------------------------------------------------------------------

    def group_candidates(
        self,
        candidates: List[ExperienceCandidate],
    ) -> List[List[ExperienceCandidate]]:
        """
        Group candidates into provisional experience groups.

        Candidates with strong separating evidence start a new group.
        """

        if not candidates:
            return []

        groups: List[List[ExperienceCandidate]] = [
            [candidates[0]]
        ]

        for candidate in candidates[1:]:

            current_group = groups[-1]

            previous_candidate = current_group[-1]

            boundary = self.detect_boundary(
                previous_candidate,
                candidate,
            )

            if boundary.score >= 0.80:
                groups.append([candidate])
            else:
                current_group.append(candidate)

        return groups

    # ------------------------------------------------------------------------
    # Segment creation
    # ------------------------------------------------------------------------

    def _create_segment(
        self,
        candidates: List[ExperienceCandidate],
        index: int,
    ) -> ExperienceSegment:
        """
        Convert one candidate group into a provisional segment.

        A segment represents one candidate-derived experience episode.
        Segmentation is still provisional; final validation happens later.
        """

        if not candidates:
            raise ValueError(
                "Cannot create an ExperienceSegment from an empty candidate group."
            )

        # ------------------------------------------------------------
        # Select the primary candidate represented by this group.
        #
        # The method receives ONE GROUP, not the complete candidate list.
        # Therefore the first candidate is the primary candidate.
        # ------------------------------------------------------------

        candidate = candidates[0]

        # ------------------------------------------------------------
        # Segment identity
        # ------------------------------------------------------------

        segment_id = f"segment-{index + 1}"

        # ------------------------------------------------------------
        # Collect all block IDs belonging to this candidate group.
        # Preserve order and avoid duplicates.
        # ------------------------------------------------------------

        block_ids: List[str] = []

        for group_candidate in candidates:
            for block_id in group_candidate.block_ids:
                if block_id not in block_ids:
                    block_ids.append(block_id)

        # ------------------------------------------------------------
        # Anchor blocks
        # ------------------------------------------------------------

        anchor_block_ids: List[str] = []

        for group_candidate in candidates:

            if not group_candidate.block_ids:
                continue

            if (
                group_candidate.has_role()
                or group_candidate.has_company()
                or group_candidate.has_date()
            ):
                for block_id in group_candidate.block_ids:
                    if block_id not in anchor_block_ids:
                        anchor_block_ids.append(block_id)

        # ------------------------------------------------------------
        # Provisional confidence
        #
        # This is NOT final confidence.
        # ------------------------------------------------------------

        confidence = 0.0

        if anchor_block_ids:
            confidence = 0.70

        if len(candidates) > 1:
            confidence = min(
                0.85,
                confidence + 0.05,
            )

        # ------------------------------------------------------------
        # Create provisional segment
        # ------------------------------------------------------------

        segment = ExperienceSegment(
            segment_id=segment_id,

            block_ids=block_ids,

            candidate_id=candidate.candidate_id,

            role=candidate.role,
            company=candidate.company,
            location=candidate.location,

            start_date=candidate.start_date,
            end_date=candidate.end_date,

            is_current=(
                candidate.is_current
                if candidate.is_current is not None
                else False
            ),

            anchor_block_ids=anchor_block_ids,

            confidence=confidence,

            status=SegmentStatus.PROVISIONAL,
        )

        return segment

    # ------------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------------

    def segment(
        self,
        candidates: List[ExperienceCandidate],
    ) -> SegmentationResult:
        """
        Segment candidates into provisional experience groups.
        """

        if not candidates:
            return SegmentationResult()

        groups = self.group_candidates(
            candidates
        )

        segments: List[ExperienceSegment] = []
        boundaries: List[BoundaryEvidence] = []

        for index, group in enumerate(groups):

            segment = self._create_segment(
                group,
                index,
            )

            # Boundary before this segment.
            if index > 0:

                previous_group = groups[index - 1]

                boundary = self.detect_boundary(
                    previous_group[-1],
                    group[0],
                )

                segment.boundary_before = boundary

                if segments:
                    segments[-1].boundary_after = boundary

                boundaries.append(boundary)

            segments.append(segment)

        return SegmentationResult(
            segments=segments,
            boundaries=boundaries,
            metadata={
                "candidate_count": len(candidates),
                "segment_count": len(segments),
            },
        )