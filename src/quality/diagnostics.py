"""Canonical document extraction quality models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QualityIssue(BaseModel):
    """A single extraction or document-quality issue."""

    severity: str
    category: str
    message: str
    suggestion: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class PageQuality(BaseModel):
    """Quality information for one document page."""

    page: int

    text_length: int = 0
    block_count: int = 0

    text_density: float = 0.0

    has_text: bool = False
    likely_scanned: bool = False
    likely_empty: bool = False

    confidence: float = 1.0

    issues: List[QualityIssue] = Field(
        default_factory=list
    )


class DocumentQualityReport(BaseModel):
    """Canonical quality report for the extracted document."""

    overall_confidence: float = 1.0

    extraction_method: str = "unknown"

    page_count: int = 0
    block_count: int = 0

    total_text_length: int = 0

    text_coverage: float = 0.0

    requires_ocr: bool = False

    has_layout_anomalies: bool = False
    has_encoding_issues: bool = False
    has_duplicate_content: bool = False

    pages: List[PageQuality] = Field(
        default_factory=list
    )

    warnings: List[QualityIssue] = Field(
        default_factory=list
    )

    errors: List[QualityIssue] = Field(
        default_factory=list
    )

    metrics: Dict[str, Any] = Field(
        default_factory=dict
    )

    def add_warning(
        self,
        issue: QualityIssue,
    ) -> None:
        self.warnings.append(issue)

    def add_error(
        self,
        issue: QualityIssue,
    ) -> None:
        self.errors.append(issue)