"""Production-oriented ATS analysis engine.

The analyzer separates:

1. Evidence collection
2. Skill matching
3. Section analysis
4. Experience-depth analysis
5. Risk detection
6. Score calculation

The current ATS score formula is intentionally preserved.
"""

import os
import re

import yaml

from typing import Dict, Any, List, Optional, Tuple

from src.document_models import (
    CanonicalResume,
    ATSAnalysisResult,
    ParsingRisk,
)
from src.analysis.skill_gap import calculate_skill_gap

from src.extraction.skills import SKILL_ALIASES

class ATSAnalyzer:

    def __init__(
        self,
        roles_config_path: str = "config/job_roles.yaml",
    ):
        self.roles_config = self._load_roles(
            roles_config_path
        )

    # =========================================================
    # CONFIGURATION
    # =========================================================

    def _load_roles(
        self,
        path: str,
    ) -> Dict[str, Any]:

        if os.path.exists(path):

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                data = yaml.safe_load(f) or {}

                return data.get(
                    "roles",
                    {},
                )

        return {}

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(
        value: Optional[str],
    ) -> str:
        """
        Normalize text for evidence storage.

        Preserves original casing so that extracted ATS evidence
        remains human-readable while normalizing whitespace.
        """

        if not value:
            return ""

        return " ".join(
            str(value).split()
        )

    @staticmethod
    def _normalize_key(
        value: str,
    ) -> str:
        """
        Normalize skill names for comparison.

        Uses the same canonical alias vocabulary as the
        technical SkillExtractor.
        """

        if not value:
            return ""

        normalized = re.sub(
            r"\s+",
            " ",
            value.lower().strip(),
        )

        # Convert known aliases to canonical skill names.
        normalized = SKILL_ALIASES.get(
            normalized,
            normalized,
        )

        # Canonical names are normalized again so comparisons
        # remain case-insensitive and whitespace-safe.
        return re.sub(
            r"\s+",
            " ",
            normalized.lower().strip(),
        )


    # =========================================================
    # SKILL MATCHING
    # =========================================================

    @staticmethod
    def _contains_term(
        text: str,
        term: str,
    ) -> bool:

        text = text or ""
        term = term.strip()

        if not text or not term:
            return False

        pattern = (
            r"(?<!\w)"
            + re.escape(term)
            + r"(?!\w)"
        )

        matched = bool(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
        )

        return matched

    # =========================================================
    # EVIDENCE CORPUS
    # =========================================================

    def _build_evidence_corpus(
        self,
        resume: CanonicalResume,
    ) -> List[Tuple[str, str]]:
        """Build one normalized evidence corpus.

        Every item is:

            (source, text)

        Example:

            (
                "experience[0].technologies",
                "Python, PyTorch, FastAPI"
            )

        The corpus intentionally contains structured evidence from
        the entire CanonicalResume instead of relying only on the
        flattened `resume.skills` list.
        """

        corpus: List[
            Tuple[str, str]
        ] = []

        # -----------------------------------------------------
        # Contact / identity
        # -----------------------------------------------------

        if resume.contact.name:
            corpus.append(
                (
                    "contact.name",
                    self._normalize_text(
                        resume.contact.name
                    ),
                )
            )

        if resume.contact.location:
            corpus.append(
                (
                    "contact.location",
                    self._normalize_text(
                        resume.contact.location
                    ),
                )
            )

        # -----------------------------------------------------
        # Profile
        # -----------------------------------------------------

        if resume.current_title:
            corpus.append(
                (
                    "profile.title",
                    self._normalize_text(
                        resume.current_title
                    ),
                )
            )

        if resume.domain_category:
            corpus.append(
                (
                    "profile.domain",
                    self._normalize_text(
                        resume.domain_category
                    ),
                )
            )

        if resume.summary:
            corpus.append(
                (
                    "summary",
                    self._normalize_text(
                        resume.summary
                    ),
                )
            )

        # -----------------------------------------------------
        # Explicit skills
        # -----------------------------------------------------

        for index, skill in enumerate(
            resume.skills
        ):

            skill_text = self._normalize_text(
                skill
            )

            if skill_text:

                corpus.append(
                    (
                        f"skills[{index}]",
                        skill_text,
                    )
                )

        # -----------------------------------------------------
        # Experience
        # -----------------------------------------------------

        for index, experience in enumerate(
            resume.experience
        ):

            prefix = f"experience[{index}]"

            if experience.role:
                corpus.append(
                    (
                        f"{prefix}.role",
                        self._normalize_text(
                            experience.role
                        ),
                    )
                )

            if experience.company:
                corpus.append(
                    (
                        f"{prefix}.company",
                        self._normalize_text(
                            experience.company
                        ),
                    )
                )

            if experience.location:
                corpus.append(
                    (
                        f"{prefix}.location",
                        self._normalize_text(
                            experience.location
                        ),
                    )
                )

            if experience.description:
                corpus.append(
                    (
                        f"{prefix}.description",
                        self._normalize_text(
                            experience.description
                        ),
                    )
                )

            # IMPORTANT:
            # Previously ignored by ATS.
            for bullet_index, bullet in enumerate(
                experience.bullet_points
            ):

                bullet_text = self._normalize_text(
                    bullet
                )

                if bullet_text:

                    corpus.append(
                        (
                            f"{prefix}.bullet[{bullet_index}]",
                            bullet_text,
                        )
                    )

            # IMPORTANT:
            # Previously ignored by ATS.
            for tech_index, technology in enumerate(
                experience.technologies_mentioned
            ):

                technology_text = (
                    self._normalize_text(
                        technology
                    )
                )

                if technology_text:

                    corpus.append(
                        (
                            f"{prefix}.technologies[{tech_index}]",
                            technology_text,
                        )
                    )

        # -----------------------------------------------------
        # Education
        # -----------------------------------------------------

        for index, education in enumerate(
            resume.education
        ):

            prefix = f"education[{index}]"

            if education.degree:
                corpus.append(
                    (
                        f"{prefix}.degree",
                        self._normalize_text(
                            education.degree
                        ),
                    )
                )

            if education.institution:
                corpus.append(
                    (
                        f"{prefix}.institution",
                        self._normalize_text(
                            education.institution
                        ),
                    )
                )

            if education.location:
                corpus.append(
                    (
                        f"{prefix}.location",
                        self._normalize_text(
                            education.location
                        ),
                    )
                )

            for detail_index, detail in enumerate(
                education.details
            ):

                detail_text = self._normalize_text(
                    detail
                )

                if detail_text:

                    corpus.append(
                        (
                            f"{prefix}.details[{detail_index}]",
                            detail_text,
                        )
                    )

        # -----------------------------------------------------
        # Projects
        # -----------------------------------------------------

        for index, project in enumerate(
            resume.projects
        ):

            prefix = f"project[{index}]"

            if project.title:
                corpus.append(
                    (
                        f"{prefix}.title",
                        self._normalize_text(
                            project.title
                        ),
                    )
                )

            if project.description:
                corpus.append(
                    (
                        f"{prefix}.description",
                        self._normalize_text(
                            project.description
                        ),
                    )
                )

            # IMPORTANT:
            # Project technologies are ATS evidence.
            for tech_index, technology in enumerate(
                project.technologies
            ):

                technology_text = (
                    self._normalize_text(
                        technology
                    )
                )

                if technology_text:

                    corpus.append(
                        (
                            f"{prefix}.technologies[{tech_index}]",
                            technology_text,
                        )
                    )

        # -----------------------------------------------------
        # Certifications
        # -----------------------------------------------------

        for index, certification in enumerate(
            resume.certifications
        ):

            prefix = f"certification[{index}]"

            if certification.name:
                corpus.append(
                    (
                        f"{prefix}.name",
                        self._normalize_text(
                            certification.name
                        ),
                    )
                )

            if certification.issuer:
                corpus.append(
                    (
                        f"{prefix}.issuer",
                        self._normalize_text(
                            certification.issuer
                        ),
                    )
                )

        # -----------------------------------------------------
        # Publications
        # -----------------------------------------------------

        for index, publication in enumerate(
            resume.publications
        ):

            publication_text = (
                self._normalize_text(
                    publication
                )
            )

            if publication_text:

                corpus.append(
                    (
                        f"publication[{index}]",
                        publication_text,
                    )
                )

        # -----------------------------------------------------
        # Awards
        # -----------------------------------------------------

        for index, award in enumerate(
            resume.awards
        ):

            award_text = self._normalize_text(
                award
            )

            if award_text:

                corpus.append(
                    (
                        f"award[{index}]",
                        award_text,
                    )
                )

        # -----------------------------------------------------
        # Achievements
        # -----------------------------------------------------

        for index, achievement in enumerate(
            resume.achievements
        ):

            achievement_text = (
                self._normalize_text(
                    achievement
                )
            )

            if achievement_text:

                corpus.append(
                    (
                        f"achievement[{index}]",
                        achievement_text,
                    )
                )

        # -----------------------------------------------------
        # Volunteer experience
        # -----------------------------------------------------

        for index, volunteer in enumerate(
            resume.volunteer_experience
        ):

            volunteer_text = (
                self._normalize_text(
                    volunteer
                )
            )

            if volunteer_text:

                corpus.append(
                    (
                        f"volunteer[{index}]",
                        volunteer_text,
                    )
                )

        return [
            (
                source,
                text,
            )
            for source, text in corpus
            if text
        ]

    # =========================================================
    # EVIDENCE EXTRACTION
    # =========================================================

    def _find_skill_evidence(
        self,
        skill: str,
        corpus,
    ) -> List[str]:
        """Return human-readable evidence for one skill."""

        evidence: List[str] = []

        if isinstance(corpus, dict):

            for source, text in corpus.items():

                if not text:
                    continue

                if self._contains_term(text, skill):
                    evidence.append(
                        f"{source}: {text}"
                    )

        elif isinstance(corpus, list):

            for item in corpus:

                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                ):
                    source, text = item

                    if not text:
                        continue

                    if self._contains_term(text, skill):
                        evidence.append(
                            f"{source}: {text}"
                        )

        return evidence

    # =========================================================
    # EVIDENCE CONFIDENCE
    # =========================================================

    @staticmethod
    def _evidence_confidence(
        evidence: List[str],
    ) -> float:
        """Estimate confidence from evidence source quality.

        This is NOT the ATS score.

        Explicit skills and structured technologies are stronger
        signals than incidental text mentions.
        """

        if not evidence:
            return 0.0

        best_confidence = 0.0

        for item in evidence:

            source = item.split(
                ":",
                1,
            )[0].lower()

            if source.startswith(
                "skills["
            ):
                confidence = 1.00

            elif ".technologies[" in source:
                confidence = 0.98

            elif ".role" in source:
                confidence = 0.95

            elif ".bullet[" in source:
                confidence = 0.93

            elif ".description" in source:
                confidence = 0.90

            elif source == "summary":
                confidence = 0.88

            elif ".title" in source:
                confidence = 0.86

            elif source.startswith(
                "certification["
            ):
                confidence = 0.84

            elif source.startswith(
                "project["
            ):
                confidence = 0.82

            else:
                confidence = 0.75

            best_confidence = max(
                best_confidence,
                confidence,
            )

        return round(
            best_confidence,
            2,
        )

    # =========================================================
    # ROLE RESOLUTION
    # =========================================================

    def resolve_role_id(
        self,
        resume: CanonicalResume,
        target_role_id: Optional[str] = None,
    ) -> Optional[str]:

        if (
            target_role_id
            and target_role_id in self.roles_config
        ):
            return target_role_id

        title = (
            resume.current_title
            or ""
        ).lower().strip()

        domain = (
            resume.domain_category
            or ""
        ).lower().strip()

        experience_text = " ".join(
            (
                f"{e.role or ''} "
                f"{e.company or ''} "
                f"{e.description or ''} "
                f"{' '.join(e.bullet_points or [])} "
                f"{' '.join(e.technologies_mentioned or [])}"
            )
            for e in resume.experience
        ).lower()

        skills_text = " ".join(
            resume.skills
        ).lower()

        full_text = (
            f"{title} "
            f"{domain} "
            f"{experience_text} "
            f"{skills_text}"
        )

        title_mappings = [
            ("mlops", "mlops_engineer"),
            ("generative ai", "genai_engineer"),
            ("genai", "genai_engineer"),
            ("machine learning engineer", "ml_engineer"),
            ("data scientist", "data_scientist"),
            ("data analyst", "data_analyst"),
            ("data engineer", "data_engineer"),
            ("ai engineer", "ai_engineer"),
            ("software engineer", "software_engineer"),
        ]

        for phrase, role_id in title_mappings:

            if phrase in title:
                return role_id

        content_mappings = [
            (
                "mlops_engineer",
                [
                    "kubernetes",
                    "mlflow",
                    "model monitoring",
                    "model deployment",
                    "ci/cd",
                    "docker",
                    "prometheus",
                    "grafana",
                ],
            ),
            (
                "genai_engineer",
                [
                    "rag",
                    "langchain",
                    "langgraph",
                    "llms",
                    "vector databases",
                    "prompt engineering",
                    "embeddings",
                ],
            ),
            (
                "data_engineer",
                [
                    "pyspark",
                    "airflow",
                    "data pipelines",
                    "data warehousing",
                    "kafka",
                    "etl",
                ],
            ),
            (
                "ai_engineer",
                [
                    "computer vision",
                    "nlp",
                    "transformers",
                    "deep learning",
                    "model deployment",
                    "hugging face",
                ],
            ),
            (
                "ml_engineer",
                [
                    "pytorch",
                    "tensorflow",
                    "model serving",
                    "model optimization",
                    "feature engineering",
                ],
            ),
            (
                "data_analyst",
                [
                    "power bi",
                    "tableau",
                    "dashboarding",
                    "excel",
                    "business intelligence",
                ],
            ),
        ]

        best_role = None
        best_score = 0

        for role_id, signals in content_mappings:

            score = sum(
                1
                for signal in signals
                if signal in full_text
            )

            if score > best_score:
                best_score = score
                best_role = role_id

        if best_role:
            return best_role

        domain_mappings = {
            "data science": "data_scientist",
            "machine learning": "ml_engineer",
            "artificial intelligence": "ai_engineer",
            "generative ai": "genai_engineer",
            "data engineering": "data_engineer",
            "data analytics": "data_analyst",
            "mlops": "mlops_engineer",
            "software engineering": "software_engineer",
        }

        return domain_mappings.get(
            domain
        )

    # =========================================================
    # SUPPORTED ROLES
    # =========================================================

    def get_supported_roles(
        self,
    ) -> List[Dict[str, str]]:

        result = []

        for key, value in self.roles_config.items():

            result.append(
                {
                    "id": key,
                    "title": value.get(
                        "title",
                        key,
                    ),
                    "category": value.get(
                        "category",
                        "Technical",
                    ),
                }
            )

        return result

    def _collect_resume_evidence(
        self,
        resume: CanonicalResume
    ) -> Dict[str, str]:
        """
        Collect searchable evidence from all meaningful resume sections.

        Evidence is intentionally structured by source so ATS matching can
        distinguish where a competency was found.
        """

        evidence: Dict[str, List[str]] = {
            "summary": [],
            "skills": [],
            "experience": [],
            "experience_technologies": [],
            "projects": [],
            "project_technologies": [],
            "certifications": [],
            "education": [],
            "publications": [],
            "achievements": [],
            "awards": [],
            "volunteer": [],
        }

        # Summary
        if resume.summary:
            evidence["summary"].append(resume.summary)

        # Explicit skills
        evidence["skills"].extend(
            str(skill)
            for skill in resume.skills
            if skill
        )

        # Experience
        for exp in resume.experience:
            if exp.role:
                evidence["experience"].append(exp.role)

            if exp.company:
                evidence["experience"].append(exp.company)

            if exp.description:
                evidence["experience"].append(exp.description)

            evidence["experience"].extend(
                bullet
                for bullet in exp.bullet_points
                if bullet
            )

            evidence["experience_technologies"].extend(
                tech
                for tech in exp.technologies_mentioned
                if tech
            )

        # Projects
        for project in resume.projects:
            if project.title:
                evidence["projects"].append(project.title)

            if project.description:
                evidence["projects"].append(project.description)

            evidence["project_technologies"].extend(
                tech
                for tech in project.technologies
                if tech
            )

        # Certifications
        for cert in resume.certifications:
            if cert.name:
                evidence["certifications"].append(cert.name)

            if cert.issuer:
                evidence["certifications"].append(cert.issuer)

        # Education
        for edu in resume.education:
            if edu.degree:
                evidence["education"].append(edu.degree)

            if edu.institution:
                evidence["education"].append(edu.institution)

            evidence["education"].extend(
                detail
                for detail in edu.details
                if detail
            )

        # Additional professional evidence
        evidence["publications"].extend(
            item for item in resume.publications if item
        )

        evidence["achievements"].extend(
            item for item in resume.achievements if item
        )

        evidence["awards"].extend(
            item for item in resume.awards if item
        )

        evidence["volunteer"].extend(
            item for item in resume.volunteer_experience if item
        )

        # Convert each evidence category to normalized searchable text.
        return {
            source: self._normalize_text(" ".join(values))
            for source, values in evidence.items()
        }

    def _build_searchable_evidence(
        self,
        resume: CanonicalResume
    ) -> str:
        """
        Build a single normalized evidence corpus for ATS matching.
        """

        evidence = self._collect_resume_evidence(resume)

        return " ".join(
            value
            for value in evidence.values()
            if value
        )

    @staticmethod
    def _skill_in_text(
        skill: str,
        text: str
    ) -> bool:
        """
        Perform safe case-insensitive phrase matching.

        Uses word boundaries to avoid accidental substring matches such as:
        'R' matching unrelated words.
        """

        if not skill or not text:
            return False

        pattern = rf"(?<!\w){re.escape(skill.lower())}(?!\w)"

        return bool(re.search(pattern, text, re.IGNORECASE))

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        resume: CanonicalResume,
        target_role_id: Optional[str] = None
    ) -> ATSAnalysisResult:
        """
        Analyze a resume against the resolved target role.

        The ATS analysis uses structured evidence collected from:
        - Summary
        - Explicit skills
        - Experience roles and descriptions
        - Experience technologies
        - Projects and project technologies
        - Certifications
        - Education
        - Publications
        - Achievements
        - Awards
        - Volunteer experience

        The scoring remains transparent and deterministic.
        """

        # ==============================================================
        # 0. RESOLVE TARGET ROLE
        # ==============================================================

        # Determine the target role using the following priority:
        # 1. Explicit user-selected role
        # 2. Resume title
        # 3. Resume content
        # 4. Resume domain
        resolved_role_id = self.resolve_role_id(
            resume,
            target_role_id=target_role_id
        )

        role_data = (
            self.roles_config.get(resolved_role_id, {})
            if resolved_role_id
            else {}
        )

        if role_data:
            target_title = role_data.get(
                "title",
                resolved_role_id.replace("_", " ").title()
            )

            core_reqs = role_data.get("core_skills", [])
            secondary_reqs = role_data.get("secondary_skills", [])
        else:
            target_title = resume.current_title
            core_reqs = []
            secondary_reqs = []

        # ==============================================================
        # 1. SECTION COMPLETENESS CHECK
        # ==============================================================

        found_sections = []
        missing_sections = []

        standard_sections = [
            "contact",
            "experience",
            "education",
            "skills",
            "projects"
        ]

        # Contact
        if resume.contact.email:
            found_sections.append("contact")
        else:
            missing_sections.append("contact")

        # Experience
        if resume.experience:
            found_sections.append("experience")
        else:
            missing_sections.append("experience")

        # Education
        if resume.education:
            found_sections.append("education")
        else:
            missing_sections.append("education")

        # Skills
        if resume.skills:
            found_sections.append("skills")
        else:
            missing_sections.append("skills")

        # Projects
        if resume.projects:
            found_sections.append("projects")
        else:
            missing_sections.append("projects")

        section_score = (
            len(found_sections) / len(standard_sections)
        ) * 100.0

        # ==============================================================
        # 2. ATS EVIDENCE COLLECTION
        # ==============================================================

        # Build structured evidence from the complete CanonicalResume.
        #
        # Each item preserves its exact source:
        #
        #   ("skills[0]", "Python")
        #   ("experience[0].technologies[0]", "PyTorch")
        #   ("experience[0].bullet[0]", "Built CNN models")
        #
        # This is required for explainable ATS evidence
        # and evidence-confidence scoring.

        evidence = self._build_evidence_corpus(
            resume
        )

        all_resume_text = " ".join(
            text
            for _, text in evidence
        )

        # Explicitly declared skills receive the strongest evidence signal.
        #
        # Normalize only for comparison. The original evidence remains
        # unchanged inside the evidence corpus.
        explicit_skills = {
            self._normalize_key(skill)
            for skill in resume.skills
            if skill
        }

        # ==============================================================
        # 2A. CORE SKILL DETECTION
        # ==============================================================

        detected_core = []
        missing_core = []
        core_skill_evidence = {}
        missing_core_skill_evidence = {}

        for required_skill in core_reqs:

            normalized_skill = self._normalize_key(
                required_skill
            )

            skill_evidence = self._find_skill_evidence(
                normalized_skill,
                evidence
            )

            if (
                normalized_skill in explicit_skills
                or self._skill_in_text(
                    normalized_skill,
                    all_resume_text
                )
            ):
                detected_core.append(
                    required_skill
                )

                core_skill_evidence[required_skill] = skill_evidence

            else:
                missing_core.append(
                    required_skill
                )

                missing_core_skill_evidence[required_skill] = []

        # ==============================================================
        # 2B. SECONDARY SKILL DETECTION
        # ==============================================================

        detected_sec = []
        missing_sec = []

        secondary_skill_evidence = {}
        missing_secondary_skill_evidence = {}

        for required_skill in secondary_reqs:

            normalized_skill = self._normalize_key(
                required_skill
            )

            skill_evidence = self._find_skill_evidence(
                normalized_skill,
                evidence
            )

            # Explicit skill is also valid evidence
            if normalized_skill in explicit_skills:

                explicit_evidence = (
                    f"Explicit skill: {required_skill}"
                )

                if explicit_evidence not in skill_evidence:
                    skill_evidence.insert(
                        0,
                        explicit_evidence
                    )

            # ----------------------------------------------------------
            # Matched / Missing
            # ----------------------------------------------------------

            if skill_evidence:

                detected_sec.append(
                    required_skill
                )

                secondary_skill_evidence[
                    required_skill
                ] = skill_evidence

            else:

                missing_sec.append(
                    required_skill
                )

                missing_secondary_skill_evidence[
                    required_skill
                ] = []

        # ==============================================================
        # 2B.5. SKILL EVIDENCE CONFIDENCE
        # ==============================================================

        skill_evidence_confidence = {}

        # Core skill confidence
        for skill, evidence_items in core_skill_evidence.items():

            skill_evidence_confidence[skill] = (
                self._evidence_confidence(
                    evidence_items
                )
            )

        # Secondary skill confidence
        for skill, evidence_items in secondary_skill_evidence.items():

            skill_evidence_confidence[skill] = (
                self._evidence_confidence(
                    evidence_items
                )
            )        
        
        # ==============================================================
        # 2C. SKILL COVERAGE SCORE
        # ==============================================================

        core_ratio = (
            len(detected_core) / max(1, len(core_reqs))
        )

        sec_ratio = (
            len(detected_sec) / max(1, len(secondary_reqs))
        )

        # Core skills receive greater weight because they are generally
        # more important to role alignment.
        skill_score = (
            core_ratio * 75.0
        ) + (
            sec_ratio * 25.0
        )

        # ==============================================================
        # 3. EXPERIENCE DEPTH CHECK
        # ==============================================================

        exp_score = 40.0

        if resume.experience:
            # Base experience presence
            exp_score += 25.0

            # Number of resume bullets
            total_bullets = sum(
                len(e.bullet_points)
                for e in resume.experience
            )

            if total_bullets >= 4:
                exp_score += 15.0

            # Detect quantified achievements.
            #
            # Current logic preserves the existing scoring behavior.
            has_metrics = False

            for e in resume.experience:
                if any(
                    char in e.description
                    for char in ["%", "$", "+", "x"]
                ):
                    has_metrics = True
                    break

            if has_metrics:
                exp_score += 20.0

        exp_score = min(100.0, exp_score)

        # ==============================================================
        # 4. PARSING RISK DETECTION
        # ==============================================================

        risks: List[ParsingRisk] = []
        recommendations: List[str] = []

        # --------------------------------------------------------------
        # Missing email
        # --------------------------------------------------------------

        if not resume.contact.email:
            risks.append(
                ParsingRisk(
                    level="HIGH",
                    category="Contact",
                    message="No direct email address found in the header.",
                    suggestion=(
                        "Include an email address at the very top "
                        "of your resume."
                    )
                )
            )

            recommendations.append(
                "Add your primary email address prominently in the header."
            )

        # --------------------------------------------------------------
        # Missing phone
        # --------------------------------------------------------------

        if not resume.contact.phone:
            risks.append(
                ParsingRisk(
                    level="MEDIUM",
                    category="Contact",
                    message="Phone number could not be verified.",
                    suggestion=(
                        "Add a formatted contact phone number."
                    )
                )
            )

        # --------------------------------------------------------------
        # Missing professional links
        # --------------------------------------------------------------

        if (
            not resume.contact.linkedin
            and not resume.contact.github
        ):
            risks.append(
                ParsingRisk(
                    level="MEDIUM",
                    category="Links",
                    message="No professional profile link was detected.",
                    suggestion=(
                        "Consider adding a relevant professional "
                        "profile or portfolio link."
                    )
                )
            )

            recommendations.append(
                "Add a relevant professional profile or portfolio link."
            )

        # --------------------------------------------------------------
        # Missing core competencies
        # --------------------------------------------------------------

        if role_data and len(missing_core) > 0:
            top_missing = ", ".join(
                missing_core[:4]
            )

            recommendations.append(
                f"Consider demonstrating experience with key "
                f"{target_title} competencies: {top_missing}."
            )

        # --------------------------------------------------------------
        # Weak experience evidence
        # --------------------------------------------------------------

        if exp_score < 70.0:
            recommendations.append(
                "Enhance experience descriptions with quantified impact, "
                "such as percentages, revenue, cost savings, growth, "
                "efficiency, or other measurable outcomes."
            )

        # ==============================================================
        # 5. OVERALL ATS SCORE
        # ==============================================================

        # Current transparent weighting:
        #
        # Section Completeness = 30%
        # Skill Coverage       = 40%
        # Experience Depth     = 25%
        # Formatting/Contact   = 5%
        #
        # Email absence creates an additional 10-point deduction.

        contact_deduction = (
            10.0
            if not resume.contact.email
            else 0.0
        )

        overall = (
            (section_score * 0.30)
            + (skill_score * 0.40)
            + (exp_score * 0.25)
            + 5.0
            - contact_deduction
        )

        # Keep the score within the existing product range.
        overall = round(
            max(
                10.0,
                min(99.0, overall)
            ),
            1
        )

        skill_gap_result = calculate_skill_gap(
            matched_core=detected_core,
            matched_secondary=detected_sec,
            missing_core=missing_core,
            missing_secondary=missing_sec,
        )

        # ==============================================================
        # 6. RETURN STRUCTURED ATS RESULT
        # ==============================================================

        return ATSAnalysisResult(
            target_role=target_title,

            overall_score=overall,

            section_completeness_score=round(
                section_score,
                1
            ),

            skill_coverage_score=round(
                skill_score,
                1
            ),

            experience_depth_score=round(
                exp_score,
                1
            ),

            detected_core_skills=detected_core,

            detected_secondary_skills=detected_sec,

            missing_core_skills=missing_core,

            missing_secondary_skills=missing_sec,

            # =========================================================
            # SKILL EVIDENCE
            # =========================================================

            core_skill_evidence=core_skill_evidence,

            secondary_skill_evidence=secondary_skill_evidence,

            missing_core_skill_evidence=missing_core_skill_evidence,

            missing_secondary_skill_evidence=missing_secondary_skill_evidence,

            skill_evidence_confidence=skill_evidence_confidence,

            # =========================================================

            skill_gap=skill_gap_result.__dict__,

            sections_found=found_sections,

            sections_missing=missing_sections,

            parsing_risks=risks,

            actionable_recommendations=recommendations
        )