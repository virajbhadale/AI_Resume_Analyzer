"""
Evidence models for universal resume intelligence.

This module contains structured representations of evidence produced
during resume inference.

Important design principle
--------------------------
Detection and decision-making are separate.

A detector should be able to say:

    "This text may be a ROLE with confidence 0.82"

without immediately deciding:

    "This IS the ROLE."

Evidence is later combined by candidate generation, evidence fusion,
validation, and resolution stages.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Evidence categories
# ============================================================================


class EvidenceType(str, Enum):
    """
    High-level source of evidence.

    These categories are intentionally generic so the same evidence
    architecture can be used for:

        - experience
        - education
        - projects
        - certifications
        - skills
        - publications
        - other resume entities
    """

    LEXICAL = "lexical"
    TYPOGRAPHY = "typography"
    LAYOUT = "layout"
    SECTION = "section"
    DATE = "date"
    ROLE = "role"
    COMPANY = "company"
    LOCATION = "location"
    SEMANTIC = "semantic"
    NEIGHBOR = "neighbor"
    STRUCTURAL = "structural"
    PROVENANCE = "provenance"


# ============================================================================
# Evidence direction
# ============================================================================


class EvidencePolarity(str, Enum):
    """
    Indicates whether a signal supports or contradicts a hypothesis.
    """

    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


# ============================================================================
# Evidence signal
# ============================================================================


class EvidenceSignal(BaseModel):
    """
    One atomic piece of evidence.

    Example
    -------
    A block contains:

        Data Analyst Intern

    The system may produce:

        signal_type = ROLE
        value = "Data Analyst Intern"
        confidence = 0.94
        polarity = SUPPORT

    Another signal might say:

        the same text appears inside a bullet

    which could become:

        signal_type = STRUCTURAL
        polarity = CONTRADICT
        confidence = 0.80

    Neither signal makes the final decision alone.
    """

    signal_type: EvidenceType

    value: Any = None

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    polarity: EvidencePolarity = (
        EvidencePolarity.SUPPORT
    )

    source_block_id: Optional[str] = None

    source_field: Optional[str] = None

    explanation: Optional[str] = None

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================================
# Field evidence
# ============================================================================


class FieldEvidence(BaseModel):
    """
    Evidence associated with one canonical resume field.

    FieldEvidence does not represent a final decision.

    It represents the current evidence collected for a field such as:

        ROLE
        COMPANY
        DATE
        LOCATION
        DESCRIPTION

    Example
    -------
    ROLE = "AI Engineer"

    may have:

        confidence = 0.94

        signals:
            - role detector
            - experience section
            - bold typography
            - semantic similarity

    The final field value is resolved later by the inference pipeline.
    """

    field: str

    value: Any = None

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    signals: List[EvidenceSignal] = Field(
        default_factory=list
    )

    source_block_ids: List[str] = Field(
        default_factory=list
    )

    accepted: bool = False

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    def add_signal(
        self,
        signal: EvidenceSignal,
    ) -> None:
        """Attach an evidence signal to this field."""

        self.signals.append(signal)

        if signal.source_block_id:
            if signal.source_block_id not in self.source_block_ids:
                self.source_block_ids.append(
                    signal.source_block_id
                )

# ============================================================================
# Hypothesis
# ============================================================================


class EvidenceHypothesis(BaseModel):
    """
    A possible interpretation of a piece of resume content.

    Examples:

        ROLE = "Data Scientist"

        COMPANY = "Google"

        LOCATION = "Pune, India"

    Multiple hypotheses may exist simultaneously.

    This is important for ambiguous resume content.
    """

    label: str

    value: Any

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    evidence: List[EvidenceSignal] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    def add_evidence(
        self,
        signal: EvidenceSignal,
    ) -> None:
        """Attach an evidence signal to this hypothesis."""

        self.evidence.append(signal)


# ============================================================================
# Evidence collection
# ============================================================================


class EvidenceCollection(BaseModel):
    """
    Collection of evidence associated with one inference unit.

    This provides a common container for evidence fusion.
    """

    signals: List[EvidenceSignal] = Field(
        default_factory=list
    )

    hypotheses: List[EvidenceHypothesis] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    def add_signal(
        self,
        signal: EvidenceSignal,
    ) -> None:
        """Add an evidence signal."""

        self.signals.append(signal)

    def add_hypothesis(
        self,
        hypothesis: EvidenceHypothesis,
    ) -> None:
        """Add a hypothesis."""

        self.hypotheses.append(hypothesis)

    def signals_for(
        self,
        signal_type: EvidenceType,
    ) -> List[EvidenceSignal]:
        """Return all signals of a specific type."""

        return [
            signal
            for signal in self.signals
            if signal.signal_type == signal_type
        ]

# ============================================================================
# Block evidence
# ============================================================================


# ============================================================================
# Block evidence
# ============================================================================


class BlockEvidence(BaseModel):
    """
    Evidence-preserving representation of one DocumentBlock.

    BlockEvidence records observations about a source block without
    deciding whether that block belongs to an experience, education,
    project, or any other final entity.

    Design principle
    ----------------
    Preserve the evidence first. Decisions happen later.

    The model therefore retains:

        - source identity
        - raw / normalized text when available
        - document position
        - layout information
        - typography
        - section context
        - semantic metadata
        - detector observations
        - field-level confidence
        - positive / contradictory evidence
        - provenance

    This model must remain independent of ExperienceEpisode.
    """

    # ------------------------------------------------------------------
    # Source identity
    # ------------------------------------------------------------------

    block_id: str

    text: str

    raw_text: Optional[str] = None

    normalized_text: Optional[str] = None

    # ------------------------------------------------------------------
    # Document position
    # ------------------------------------------------------------------

    page: int = 1

    column_id: int = 0

    reading_order: int = 0

    bbox: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Layout / typography
    # ------------------------------------------------------------------

    layout_region: Optional[str] = None

    is_bold: bool = False

    is_heading: bool = False

    block_type: str = "text"

    # ------------------------------------------------------------------
    # Semantic / section context
    # ------------------------------------------------------------------

    section: Optional[str] = None

    section_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    semantic_label: Optional[str] = None

    # ------------------------------------------------------------------
    # Extraction provenance
    # ------------------------------------------------------------------

    source_type: str = "unknown"

    extraction_method: str = "unknown"

    detector: Optional[str] = None

    source_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    provenance: Dict[str, Any] = Field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Experience-related observations
    # ------------------------------------------------------------------

    role: Optional[str] = None

    company: Optional[str] = None

    start_date: Optional[str] = None

    end_date: Optional[str] = None

    is_current: bool = False

    location: Optional[str] = None

    # ------------------------------------------------------------------
    # Field-level confidence
    # ------------------------------------------------------------------

    role_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    company_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    date_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    location_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    signals: List[EvidenceSignal] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_role(self) -> bool:
        return bool(self.role)

    @property
    def has_company(self) -> bool:
        return bool(self.company)

    @property
    def has_date(self) -> bool:
        return bool(
            self.start_date
            or self.end_date
            or self.is_current
        )

    @property
    def has_location(self) -> bool:
        return bool(self.location)

    @property
    def is_experience_anchor(self) -> bool:
        """
        Identify whether this block contains enough positive
        experience-related evidence to participate in candidate
        generation.

        This is NOT a final experience decision.
        """

        return bool(
            self.has_role
            and (
                self.has_date
                or self.has_company
                or self.has_location
            )
        )

    
# ============================================================================
# Resolution result
# ============================================================================


class EvidenceResolution(BaseModel):
    """
    Result produced after evidence fusion and validation.

    This represents the system's current best interpretation while
    retaining the evidence that led to the decision.
    """

    label: str

    value: Any = None

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    accepted: bool = False

    evidence: List[EvidenceSignal] = Field(
        default_factory=list
    )

    rejection_reason: Optional[str] = None

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

# ============================================================================
# Anchor evidence
# ============================================================================


class AnchorType(str, Enum):
    """
    Classification of how strongly a group of resume blocks can act
    as an experience anchor.

    The type describes the evidence pattern, not a final decision.
    """

    ROLE_COMPANY_DATE = "role_company_date"
    ROLE_DATE = "role_date"
    ROLE_COMPANY = "role_company"
    COMPANY_DATE_LOCATION = "company_date_location"

    ROLE_LOCATION_DATE = "role_location_date"

    ROLE_ONLY = "role_only"
    COMPANY_ONLY = "company_only"

    UNKNOWN = "unknown"


class AnchorStrength(str, Enum):
    """
    Qualitative strength of an experience anchor.
    """

    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


class AnchorEvidence(BaseModel):
    """
    Evidence describing why a group of blocks may represent an
    experience anchor.

    AnchorEvidence does NOT create or confirm an ExperienceEpisode.

    It answers:

        "How strongly does this group of evidence resemble an
         experience header?"

    Final decisions are made later by candidate generation,
    segmentation, metadata validation, and confidence scoring.
    """

    anchor_type: AnchorType = AnchorType.UNKNOWN

    strength: AnchorStrength = AnchorStrength.WEAK

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    block_ids: List[str] = Field(
        default_factory=list
    )

    supporting_evidence: List[EvidenceSignal] = Field(
        default_factory=list
    )

    contradictory_evidence: List[EvidenceSignal] = Field(
        default_factory=list
    )

    explanation: Optional[str] = None

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    def add_supporting_evidence(
        self,
        signal: EvidenceSignal,
    ) -> None:
        """Add evidence supporting the anchor."""

        self.supporting_evidence.append(signal)

    def add_contradictory_evidence(
        self,
        signal: EvidenceSignal,
    ) -> None:
        """Add evidence contradicting the anchor."""

        self.contradictory_evidence.append(signal)

    def add_block(
        self,
        block_id: str,
    ) -> None:
        """Add a source block without duplication."""

        if block_id not in self.block_ids:
            self.block_ids.append(block_id)

    @property
    def is_strong(self) -> bool:
        return self.strength == AnchorStrength.STRONG

    @property
    def is_medium(self) -> bool:
        return self.strength == AnchorStrength.MEDIUM

    @property
    def is_weak(self) -> bool:
        return self.strength == AnchorStrength.WEAK 


# ============================================================================
# Boundary evidence
# ============================================================================


class BoundaryType(str, Enum):
    """
    Describes the type of possible boundary between two blocks/groups.
    """

    NEW_ROLE = "new_role"
    NEW_COMPANY = "new_company"
    NEW_DATE_RANGE = "new_date_range"

    LAYOUT_BREAK = "layout_break"
    SEMANTIC_SHIFT = "semantic_shift"

    SECTION_BREAK = "section_break"

    SAME_EXPERIENCE = "same_experience"
    DESCRIPTION_CONTINUATION = "description_continuation"

    UNKNOWN = "unknown"


class BoundaryEvidence(BaseModel):
    """
    Evidence describing whether a boundary exists between two
    resume blocks or regions.

    BoundaryEvidence does not decide segmentation.

    It records evidence that can later be used by the segmenter
    to determine whether a new ExperienceEpisode should begin.
    """

    boundary_type: BoundaryType = BoundaryType.UNKNOWN

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    previous_block_ids: List[str] = Field(
        default_factory=list
    )

    next_block_ids: List[str] = Field(
        default_factory=list
    )

    supporting_evidence: List[EvidenceSignal] = Field(
        default_factory=list
    )

    contradictory_evidence: List[EvidenceSignal] = Field(
        default_factory=list
    )

    explanation: Optional[str] = None

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    def add_previous_block(
        self,
        block_id: str,
    ) -> None:
        """Add a previous-side block without duplication."""

        if block_id not in self.previous_block_ids:
            self.previous_block_ids.append(block_id)

    def add_next_block(
        self,
        block_id: str,
    ) -> None:
        """Add a next-side block without duplication."""

        if block_id not in self.next_block_ids:
            self.next_block_ids.append(block_id)

    def add_supporting_evidence(
        self,
        signal: EvidenceSignal,
    ) -> None:
        """Add evidence supporting the existence of a boundary."""

        self.supporting_evidence.append(signal)

    def add_contradictory_evidence(
        self,
        signal: EvidenceSignal,
    ) -> None:
        """Add evidence arguing against a boundary."""

        self.contradictory_evidence.append(signal)       