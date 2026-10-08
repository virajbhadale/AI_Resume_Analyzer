from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.inference.candidates import (
    CandidateStatus,
    ExperienceCandidate,
)

from src.inference.evidence_models import (
    EvidencePolarity,
    EvidenceSignal,
)


class EvidenceCollection(list[EvidenceSignal]):
    """
    Evidence collection that preserves actual EvidenceSignal objects
    while supporting legacy string-membership checks.

    Examples
    --------
    collection[0]
        -> EvidenceSignal

    len(collection)
        -> number of actual evidence signals

    "role" in collection
        -> True when a ROLE signal exists or role is structurally present

    "missing_company" in collection
        -> True when company is structurally missing

    Structural compatibility labels are virtual. They are NOT stored
    as fabricated EvidenceSignal objects.
    """

    def __init__(
        self,
        signals: List[EvidenceSignal] | None = None,
        compatibility_labels: List[str] | None = None,
    ) -> None:
        super().__init__(signals or [])
        self._compatibility_labels = set(
            compatibility_labels or []
        )

    def __contains__(self, item: object) -> bool:
        """
        Support both structured EvidenceSignal membership and
        backward-compatible string membership.
        """

        if isinstance(item, str):
            if item in self._compatibility_labels:
                return True

            return any(
                signal.signal_type.value == item
                for signal in self
            )

        return super().__contains__(item)


@dataclass(frozen=True)
class ValidationResult:
    """
    Result produced by ExperienceCandidateValidator.

    ``positive_evidence`` and ``negative_evidence`` contain the
    original EvidenceSignal objects.

    Legacy string membership such as:

        "role" in result.positive_evidence
        "missing_company" in result.negative_evidence

    remains supported through EvidenceCollection.

    ``positive_signals`` and ``negative_signals`` explicitly expose
    the provenance-preserving EvidenceSignal lists.
    """

    status: CandidateStatus
    confidence: float
    reasons: List[str] = field(default_factory=list)

    # Evidence collections preserve actual EvidenceSignal objects
    # while supporting legacy string-membership behavior.
    positive_evidence: EvidenceCollection = field(
        default_factory=EvidenceCollection
    )
    negative_evidence: EvidenceCollection = field(
        default_factory=EvidenceCollection
    )

    # Explicit provenance-preserving signal collections.
    positive_signals: List[EvidenceSignal] = field(
        default_factory=list
    )
    negative_signals: List[EvidenceSignal] = field(
        default_factory=list
    )

    @property
    def is_valid(self) -> bool:
        """Return True only when the candidate is validated."""
        return self.status == CandidateStatus.VALIDATED


class ExperienceCandidateValidator:
    """
    Validate an ExperienceCandidate using structural and semantic
    evidence already assembled by candidate generation.

    This validator does not perform extraction, candidate generation,
    metadata resolution, or segmentation.
    """

    def validate(
        self,
        candidate: ExperienceCandidate,
    ) -> ValidationResult:
        """
        Validate one experience candidate.

        The validator consumes evidence already attached to the
        candidate. It does not generate additional evidence signals.

        Structural observations such as missing metadata are represented
        as compatibility labels and validation reasons. They are NOT
        fabricated as EvidenceSignal objects.
        """

        reasons: List[str] = []

        positive_signals, negative_signals = (
            self._analyze_evidence(candidate)
        )

        # ----------------------------------------------------------
        # 1. Structural field presence
        # ----------------------------------------------------------

        has_role = candidate.has_role()
        has_date = candidate.has_date()
        has_company = candidate.has_company()
        has_location = candidate.has_location()

        # ----------------------------------------------------------
        # 2. Backward-compatible structural labels
        #
        # These are NOT EvidenceSignal objects.
        # They exist only to preserve the historical validation API.
        # ----------------------------------------------------------

        positive_labels: List[str] = []
        negative_labels: List[str] = []

        if has_role:
            positive_labels.append("role")

        if has_date:
            positive_labels.append("date")

        if has_company:
            positive_labels.append("company")

        if not has_role:
            negative_labels.append("missing_role")

        if not has_date:
            negative_labels.append("missing_date")

        if not has_company:
            negative_labels.append("missing_company")

        # ----------------------------------------------------------
        # 3. Empty candidate
        # ----------------------------------------------------------

        if not candidate.block_ids:
            return ValidationResult(
                status=CandidateStatus.REJECTED,
                confidence=1.0,
                reasons=[
                    "Candidate contains no source blocks."
                ],
                positive_evidence=EvidenceCollection(
                    positive_signals,
                    positive_labels,
                ),
                negative_evidence=EvidenceCollection(
                    negative_signals,
                    negative_labels,
                ),
                positive_signals=positive_signals,
                negative_signals=negative_signals,
            )

        # ----------------------------------------------------------
        # 4. Validation observations
        #
        # These observations belong in reasons.
        # They are not fabricated evidence signals.
        # ----------------------------------------------------------

        if not has_role:
            reasons.append(
                "Candidate has no role evidence."
            )

        if not has_date:
            reasons.append(
                "Candidate has no date evidence."
            )

        if not has_company:
            reasons.append(
                "Candidate has no company evidence."
            )

        if has_location:
            reasons.append(
                "Candidate contains location evidence."
            )

        # ----------------------------------------------------------
        # 5. Strong experience structures
        # ----------------------------------------------------------

        role_and_date = (
            has_role
            and has_date
        )

        role_and_company = (
            has_role
            and has_company
        )

        role_company_date = (
            has_role
            and has_company
            and has_date
        )

        # ----------------------------------------------------------
        # 6. Reject candidates without meaningful experience identity
        # ----------------------------------------------------------

        if not has_role:
            return ValidationResult(
                status=CandidateStatus.REJECTED,
                confidence=0.90,
                reasons=reasons,
                positive_evidence=EvidenceCollection(
                    positive_signals,
                    positive_labels,
                ),
                negative_evidence=EvidenceCollection(
                    negative_signals,
                    negative_labels,
                ),
                positive_signals=positive_signals,
                negative_signals=negative_signals,
            )

        # ----------------------------------------------------------
        # 7. Strong validated candidate
        # ----------------------------------------------------------

        if role_company_date:
            reasons.append(
                "Candidate contains role, company, and date evidence."
            )

            return ValidationResult(
                status=CandidateStatus.VALIDATED,
                confidence=0.95,
                reasons=reasons,
                positive_evidence=EvidenceCollection(
                    positive_signals,
                    positive_labels,
                ),
                negative_evidence=EvidenceCollection(
                    negative_signals,
                    negative_labels,
                ),
                positive_signals=positive_signals,
                negative_signals=negative_signals,
            )

        # ----------------------------------------------------------
        # 8. Role + date
        # ----------------------------------------------------------

        if role_and_date:
            reasons.append(
                "Candidate contains role and date evidence."
            )

            return ValidationResult(
                status=CandidateStatus.VALIDATED,
                confidence=0.90,
                reasons=reasons,
                positive_evidence=EvidenceCollection(
                    positive_signals,
                    positive_labels,
                ),
                negative_evidence=EvidenceCollection(
                    negative_signals,
                    negative_labels,
                ),
                positive_signals=positive_signals,
                negative_signals=negative_signals,
            )

        # ----------------------------------------------------------
        # 9. Role + company
        # ----------------------------------------------------------

        if role_and_company:
            reasons.append(
                "Candidate contains role and company evidence."
            )

            return ValidationResult(
                status=CandidateStatus.VALIDATED,
                confidence=0.88,
                reasons=reasons,
                positive_evidence=EvidenceCollection(
                    positive_signals,
                    positive_labels,
                ),
                negative_evidence=EvidenceCollection(
                    negative_signals,
                    negative_labels,
                ),
                positive_signals=positive_signals,
                negative_signals=negative_signals,
            )

        # ----------------------------------------------------------
        # 10. Role only
        # ----------------------------------------------------------

        reasons.append(
            "Candidate contains role evidence but lacks "
            "supporting company or date evidence."
        )

        return ValidationResult(
            status=CandidateStatus.NEEDS_REVIEW,
            confidence=0.60,
            reasons=reasons,
            positive_evidence=EvidenceCollection(
                positive_signals,
                positive_labels,
            ),
            negative_evidence=EvidenceCollection(
                negative_signals,
                negative_labels,
            ),
            positive_signals=positive_signals,
            negative_signals=negative_signals,
        )

    def _analyze_evidence(
        self,
        candidate: ExperienceCandidate,
    ) -> tuple[
        List[EvidenceSignal],
        List[EvidenceSignal],
    ]:
        """
        Preserve the original EvidenceSignal objects.

        SUPPORT signals are positive evidence.
        CONTRADICT signals are negative evidence.
        NEUTRAL signals are ignored.

        No structural evidence is fabricated here.
        """

        positive_signals: List[EvidenceSignal] = []
        negative_signals: List[EvidenceSignal] = []

        for signal in candidate.evidence.signals:
            if signal.polarity == EvidencePolarity.SUPPORT:
                positive_signals.append(signal)

            elif signal.polarity == EvidencePolarity.CONTRADICT:
                negative_signals.append(signal)

        return positive_signals, negative_signals