from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.inference.evidence_models import EvidenceCollection


class CandidateStatus(str, Enum):
    """
    Current lifecycle state of an experience candidate.
    """

    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class ExperienceCandidate(BaseModel):
    """
    Represents a possible experience constructed from evidence.

    This is not yet a confirmed ExperienceEpisode.

    Candidate generation answers:

        "Does this collection of blocks/evidence look like
         one possible professional experience?"

    Segmentation and validation later determine whether it
    becomes a final experience episode.
    """

    candidate_id: str

    block_ids: List[str] = Field(
        default_factory=list
    )

    role: Optional[str] = None

    company: Optional[str] = None

    location: Optional[str] = None

    start_date: Optional[Any] = None

    end_date: Optional[Any] = None

    is_current: Optional[bool] = None

    section: Optional[str] = None

    # Description is optional at candidate stage. The assembled block ids
    # remain the authoritative provenance until downstream validation.

    description: Optional[str] = None

    description_block_ids: List[str] = Field(
        default_factory=list
    )

    evidence: EvidenceCollection = Field(
        default_factory=EvidenceCollection
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    status: CandidateStatus = (
        CandidateStatus.CANDIDATE
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    def add_block(
        self,
        block_id: str,
    ) -> None:
        """
        Add a block to the candidate without duplicating it.
        """

        if block_id not in self.block_ids:
            self.block_ids.append(block_id)

    def has_role(self) -> bool:
        return bool(self.role)

    def has_company(self) -> bool:
        return bool(self.company)

    def has_date(self) -> bool:
        return bool(
            self.start_date
            or self.end_date
            or self.is_current
        )

    def has_location(self) -> bool:
        return bool(self.location)

    def add_description_block(
        self,
        block_id: str,
    ) -> None:
        if block_id not in self.description_block_ids:
            self.description_block_ids.append(block_id)