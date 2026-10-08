"""
Experience inference orchestration and candidate-to-episode materialization.

This module is responsible for converting an already-built and validated
ExperienceCandidate into the existing public ExperienceEpisode model.

Important architectural boundary
---------------------------------
Candidate generation:
    DocumentBlock
        -> BlockEvidence
        -> ExperienceCandidate

Materialization:
    ExperienceCandidate
        -> validation result
        -> final ExperienceEpisode

This module does NOT:
    - detect anchors
    - generate candidates
    - segment raw document blocks
    - invent missing metadata
    - replace the existing ExperienceEpisode model
    - modify ResumeBuilder responsibilities
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.document_models import ExperienceEpisode
from src.inference.candidates import (
    CandidateStatus,
    ExperienceCandidate,
)
from src.inference.confidence import ConfidenceScorer
from src.inference.metadata import MetadataResolver
from src.inference.validation import (
    ExperienceCandidateValidator,
    ValidationResult,
)


class ExperienceEpisodeMaterializer:
    """
    Convert a validated ExperienceCandidate into an ExperienceEpisode.

    The materializer is deliberately deterministic and side-effect free
    with respect to the candidate. The candidate is never mutated.

    Missing metadata remains missing in the evidence/provenance layer.
    Existing public ExperienceEpisode sentinel values are used only because
    the current model contract requires role/company to be strings.
    """

    PROVENANCE_VERSION = 1

    ROLE_MISSING = "Role Not Specified"
    COMPANY_MISSING = "Company Not Specified"

    def __init__(
        self,
        validator: Optional[
            ExperienceCandidateValidator
        ] = None,
        metadata_resolver: Optional[
            MetadataResolver
        ] = None,
        confidence_scorer: Optional[
            ConfidenceScorer
        ] = None,
    ) -> None:
        self.validator = (
            validator
            if validator is not None
            else ExperienceCandidateValidator()
        )

        self.metadata_resolver = (
            metadata_resolver
            if metadata_resolver is not None
            else MetadataResolver()
        )

        self.confidence_scorer = (
            confidence_scorer
            if confidence_scorer is not None
            else ConfidenceScorer()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def materialize(
        self,
        candidate: ExperienceCandidate,
        validation: Optional[ValidationResult] = None,
    ) -> ExperienceEpisode:
        """
        Materialize one candidate into one final ExperienceEpisode.

        Parameters
        ----------
        candidate:
            Candidate containing resolved fields and provenance.

        validation:
            Optional previously computed validation result.

        Returns
        -------
        ExperienceEpisode
            Backward-compatible final episode containing structured
            provenance inside the existing evidence dictionary.

        Raises
        ------
        ValueError
            If the candidate is rejected or structurally invalid.
        """

        result = (
            validation
            if validation is not None
            else self.validator.validate(candidate)
        )

        if not result.is_valid:
            raise ValueError(
                "Cannot materialize rejected experience candidate "
                f"{candidate.candidate_id}: {result.reasons}"
            )

        episode = self._build_episode(candidate)

        episode.evidence = self._build_provenance(
            candidate=candidate,
            validation=result,
        )

        # Final confidence must be calculated from the final episode,
        # not copied blindly from preliminary candidate confidence.
        episode.confidence = (
            self.confidence_scorer.score_experience_episode(
                episode
            )
        )

        return episode

    # ------------------------------------------------------------------
    # Episode construction
    # ------------------------------------------------------------------

    def _build_episode(
        self,
        candidate: ExperienceCandidate,
    ) -> ExperienceEpisode:
        """
        Build the backward-compatible ExperienceEpisode representation.

        No missing metadata is invented here.
        """

        role = self._clean_optional(candidate.role)
        company = self._clean_optional(candidate.company)
        location = self._clean_optional(candidate.location)

        start_date = self._clean_optional(
            candidate.start_date
        )

        end_date = self._clean_optional(
            candidate.end_date
        )

        is_current = bool(
            candidate.is_current
            if candidate.is_current is not None
            else self._is_current_from_end_date(end_date)
        )

        metadata = candidate.metadata or {}

        description = self._extract_description(
            metadata
        )

        bullet_points = self._extract_list(
            metadata,
            "bullet_points",
        )

        technologies = self._extract_list(
            metadata,
            "technologies_mentioned",
        )

        return ExperienceEpisode(
            id=candidate.candidate_id,
            role=role or self.ROLE_MISSING,
            company=company or self.COMPANY_MISSING,
            location=location,
            start_date=start_date,
            end_date=(
                "Present"
                if is_current and not end_date
                else end_date
            ),
            is_current=is_current,
            description=description,
            bullet_points=bullet_points,
            technologies_mentioned=technologies,
            confidence=0.0,
            evidence={},
        )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _build_provenance(
        self,
        candidate: ExperienceCandidate,
        validation: ValidationResult,
    ) -> Dict[str, Any]:
        """
        Serialize candidate and validation provenance into the existing
        ExperienceEpisode.evidence dictionary.

        Only actual evidence objects are serialized.

        No synthetic evidence signals are created here.
        """

        evidence_signals = self._candidate_signals(
            candidate
        )

        supporting = [
            signal
            for signal in evidence_signals
            if self._polarity(signal) == "support"
        ]

        contradictory = [
            signal
            for signal in evidence_signals
            if self._polarity(signal) == "contradict"
        ]

        validation_supporting = list(
            getattr(
                validation,
                "positive_signals",
                [],
            )
            or []
        )

        validation_contradictory = list(
            getattr(
                validation,
                "negative_signals",
                [],
            )
            or []
        )

        return {
            "provenance_version": self.PROVENANCE_VERSION,

            "source_block_ids": list(
                candidate.block_ids
            ),

            "candidate_id": candidate.candidate_id,

            "candidate_confidence": round(
                float(candidate.confidence),
                4,
            ),

            "field_evidence": {
                "role": self._signals_for_field(
                    evidence_signals,
                    "role",
                ),
                "company": self._signals_for_field(
                    evidence_signals,
                    "company",
                ),
                "location": self._signals_for_field(
                    evidence_signals,
                    "location",
                ),
                "start_date": self._signals_for_field(
                    evidence_signals,
                    "start_date",
                ),
                "end_date": self._signals_for_field(
                    evidence_signals,
                    "end_date",
                ),
            },

            "supporting_evidence": [
                self._serialize_signal(signal)
                for signal in supporting
            ],

            "contradictory_evidence": [
                self._serialize_signal(signal)
                for signal in contradictory
            ],

            "validation": {
                "status": self._enum_value(
                    validation.status
                ),
                "confidence": round(
                    float(validation.confidence),
                    4,
                ),
                "reasons": list(
                    validation.reasons or []
                ),
                "positive_evidence": [
                    self._serialize_signal(signal)
                    for signal in validation_supporting
                ],
                "negative_evidence": [
                    self._serialize_signal(signal)
                    for signal in validation_contradictory
                ],
            },

            "metadata_resolution": {
                "role_resolved": bool(
                    self._clean_optional(candidate.role)
                ),
                "company_resolved": bool(
                    self._clean_optional(candidate.company)
                ),
                "location_resolved": bool(
                    self._clean_optional(candidate.location)
                ),
                "start_date_resolved": bool(
                    self._clean_optional(
                        candidate.start_date
                    )
                ),
                "end_date_resolved": bool(
                    self._clean_optional(
                        candidate.end_date
                    )
                ),
                "current_status_resolved": (
                    candidate.is_current is not None
                ),
            },
        }

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_signals(
        candidate: ExperienceCandidate,
    ) -> List[Any]:
        """
        Extract canonical EvidenceSignal objects from EvidenceCollection.

        The method deliberately does not assume that EvidenceCollection
        itself is iterable.
        """

        evidence = candidate.evidence

        if evidence is None:
            return []

        signals = getattr(
            evidence,
            "signals",
            None,
        )

        if signals is None:
            return []

        return list(signals)

    @classmethod
    def _signals_for_field(
        cls,
        signals: Iterable[Any],
        field_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Return serialized signals associated with a field.

        Matching uses the canonical source_field attribute where available.
        """

        result: List[Dict[str, Any]] = []

        for signal in signals:
            source_field = getattr(
                signal,
                "source_field",
                None,
            )

            if source_field != field_name:
                continue

            result.append(
                cls._serialize_signal(signal)
            )

        return result

    @staticmethod
    def _serialize_signal(
        signal: Any,
    ) -> Dict[str, Any]:
        """
        Convert an EvidenceSignal into a stable JSON-compatible dictionary.
        """

        if hasattr(signal, "model_dump"):
            data = signal.model_dump(
                mode="json"
            )
        elif hasattr(signal, "dict"):
            data = signal.dict()
        elif isinstance(signal, dict):
            data = dict(signal)
        else:
            raise TypeError(
                "Unsupported evidence signal type: "
                f"{type(signal).__name__}"
            )

        return data

    @staticmethod
    def _polarity(
        signal: Any,
    ) -> str:
        polarity = getattr(
            signal,
            "polarity",
            None,
        )

        return ExperienceEpisodeMaterializer._enum_value(
            polarity
        )

    @staticmethod
    def _enum_value(value: Any) -> Any:
        if value is None:
            return None

        return getattr(
            value,
            "value",
            value,
        )

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_optional(
        value: Any,
    ) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        return text

    @staticmethod
    def _is_current_from_end_date(
        end_date: Optional[str],
    ) -> bool:
        if not end_date:
            return False

        return end_date.strip().lower() in {
            "present",
            "current",
            "ongoing",
            "now",
            "today",
            "till date",
        }

    @staticmethod
    def _extract_description(
        metadata: Dict[str, Any],
    ) -> str:
        value = metadata.get(
            "description",
            "",
        )

        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def _extract_list(
        metadata: Dict[str, Any],
        key: str,
    ) -> List[str]:
        value = metadata.get(
            key,
            [],
        )

        if value is None:
            return []

        if isinstance(value, str):
            return [value.strip()] if value.strip() else []

        if not isinstance(value, (list, tuple)):
            return []

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]