"""Canonical Document and Resume Data Models using Pydantic."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.analysis.skill_gap import SkillGapResult
from src.analysis.skill_gap import calculate_skill_gap

class BBox(BaseModel):
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


class DocumentBlock(BaseModel):
    """
    Canonical low-level document representation.

    A DocumentBlock represents one visually meaningful piece of
    extracted document content while preserving geometry and
    provenance for downstream layout and semantic inference.

    IMPORTANT:
    This model intentionally contains NO resume-specific assumptions.
    It can represent:
        - single-column resumes
        - multi-column resumes
        - academic CVs
        - designer resumes
        - tables
        - OCR content
        - PDF text
        - DOCX content
        - unusual layouts
    """

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    id: str

    # ---------------------------------------------------------
    # Document location
    # ---------------------------------------------------------

    page: int = 1

    page_width: float = 0.0
    page_height: float = 0.0

    # ---------------------------------------------------------
    # Content
    # ---------------------------------------------------------

    text: str

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------

    bbox: BBox = Field(default_factory=BBox)

    # ---------------------------------------------------------
    # Typography
    # ---------------------------------------------------------

    font_size: float = 10.0
    font_name: str = "default"

    is_bold: bool = False
    is_italic: bool = False
    is_heading: bool = False

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------

    column_id: int = 0

    reading_order: int = 0

    # ---------------------------------------------------------
    # Provenance
    # ---------------------------------------------------------

    source_type: str = "unknown"
    """
    Extraction source.

    Examples:
        pdf
        docx
        ocr
        image
        text
    """

    extraction_method: str = "unknown"
    """
    More specific extraction method.

    Examples:
        pdf_text
        pdf_ocr
        docx_paragraph
        docx_table
        image_ocr
        text_file
    """

    # ---------------------------------------------------------
    # Structural metadata
    # ---------------------------------------------------------

    block_type: str = "text"
    """
    Generic structural type.

    Examples:
        text
        heading
        table
        table_cell
        header
        footer
        image
        list_item
        unknown
    """

    is_list_item: bool = False

    is_header_candidate: bool = False
    is_footer_candidate: bool = False

    # ---------------------------------------------------------
    # Semantic metadata
    # ---------------------------------------------------------

    section: Optional[str] = None

    semantic_label: Optional[str] = None
    """
    Examples:
        ROLE
        COMPANY
        DATE
        SKILL
        DEGREE
        CONTACT
        LOCATION
    """

    # ---------------------------------------------------------
    # Quality
    # ---------------------------------------------------------

    confidence: float = 1.0

    # ---------------------------------------------------------
    # Optional raw/provenance metadata
    # ---------------------------------------------------------

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ---------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------

    @property
    def center_x(self) -> float:
        return self.bbox.center_x

    @property
    def center_y(self) -> float:
        return self.bbox.center_y

    @property
    def width(self) -> float:
        return self.bbox.width

    @property
    def height(self) -> float:
        return self.bbox.height

    @property
    def normalized_x0(self) -> float:
        if self.page_width <= 0:
            return 0.0
        return self.bbox.x0 / self.page_width

    @property
    def normalized_x1(self) -> float:
        if self.page_width <= 0:
            return 0.0
        return self.bbox.x1 / self.page_width

    @property
    def normalized_y0(self) -> float:
        if self.page_height <= 0:
            return 0.0
        return self.bbox.y0 / self.page_height

    @property
    def normalized_y1(self) -> float:
        if self.page_height <= 0:
            return 0.0
        return self.bbox.y1 / self.page_height


class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    location: Optional[str] = None
    confidence: Dict[str, float] = Field(default_factory=dict)


class ExperienceEpisode(BaseModel):
    id: str
    role: str
    company: str
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    description: str = ""
    bullet_points: List[str] = Field(default_factory=list)
    technologies_mentioned: List[str] = Field(default_factory=list)
    confidence: float = 0.85
    evidence: Dict[str, Any] = Field(default_factory=dict)


class EducationEntry(BaseModel):
    id: str
    degree: str
    institution: str
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    details: List[str] = Field(default_factory=list)
    confidence: float = 0.85


class ProjectEntry(BaseModel):
    id: str
    title: str
    description: str = ""
    technologies: List[str] = Field(default_factory=list)
    link: Optional[str] = None
    confidence: float = 0.80


class CertificationEntry(BaseModel):
    id: str
    name: str
    issuer: Optional[str] = None
    date: Optional[str] = None
    confidence: float = 0.85


class ParsingRisk(BaseModel):
    level: str  # "LOW", "MEDIUM", "HIGH"
    category: str
    message: str
    suggestion: str


class ATSAnalysisResult(BaseModel):
    target_role: str

    # ---------------------------------------------------------
    # Overall ATS metrics
    # ---------------------------------------------------------

    overall_score: float = 0.0
    section_completeness_score: float = 0.0
    skill_coverage_score: float = 0.0
    experience_depth_score: float = 0.0
    formatting_risk_score: float = 0.0

    # ---------------------------------------------------------
    # Skill matching
    # ---------------------------------------------------------

    detected_core_skills: List[str] = Field(
        default_factory=list
    )

    detected_secondary_skills: List[str] = Field(
        default_factory=list
    )

    missing_core_skills: List[str] = Field(
        default_factory=list
    )

    missing_secondary_skills: List[str] = Field(
        default_factory=list
    )
    skill_gap: Dict[str, Any] = Field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # Explainable skill evidence
    #
    # Example:
    #
    # {
    #     "Python": [
    #         "skills: Python",
    #         "experience[0].technologies: Python"
    #     ]
    # }
    # ---------------------------------------------------------

    core_skill_evidence: Dict[str, List[str]] = Field(
        default_factory=dict
    )

    secondary_skill_evidence: Dict[str, List[str]] = Field(
        default_factory=dict
    )

    # Missing skills intentionally contain empty evidence lists.
    #
    # Example:
    #
    # {
    #     "OpenCV": []
    # }
    #
    # This makes the absence explainable instead of silently
    # returning only a keyword list.
    missing_core_skill_evidence: Dict[str, List[str]] = Field(
        default_factory=dict
    )

    missing_secondary_skill_evidence: Dict[str, List[str]] = Field(
        default_factory=dict
    )

    # Confidence of the evidence used to detect each skill.
    #
    # Example:
    #
    # {
    #     "Python": 1.0,
    #     "PyTorch": 0.95
    # }
    #
    # This is evidence confidence, NOT the ATS score.
    skill_evidence_confidence: Dict[str, float] = Field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # Section completeness
    # ---------------------------------------------------------

    sections_found: List[str] = Field(
        default_factory=list
    )

    sections_missing: List[str] = Field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Parsing / formatting risks
    # ---------------------------------------------------------

    parsing_risks: List[ParsingRisk] = Field(
        default_factory=list
    )

    actionable_recommendations: List[str] = Field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Optional Job Description match results
    # ---------------------------------------------------------

    jd_similarity_score: Optional[float] = None

    jd_matched_keywords: List[str] = Field(
        default_factory=list
    )

    jd_missing_keywords: List[str] = Field(
        default_factory=list
    )


class CanonicalResume(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)

    # Dynamically detected professional profile
    current_title: str = Field(
        default="Professional",
        description=(
            "Candidate's actual current or target professional title "
            "extracted from the resume. Do not assume a technical role."
        )
    )

    domain_category: str = Field(
        default="General",
        description=(
            "Core professional domain inferred from resume content, "
            "such as Marketing, Finance, Data Science, Software Engineering, "
            "Human Resources, Sales, etc."
        )
    )

    summary: str = ""
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceEpisode] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    projects: List[ProjectEntry] = Field(default_factory=list)
    certifications: List[CertificationEntry] = Field(default_factory=list)
    publications: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    volunteer_experience: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
