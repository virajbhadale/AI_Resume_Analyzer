"""Production-oriented Job Description Matching Engine."""

import re
from typing import Dict, Any, List, Tuple

from src.document_models import CanonicalResume
from src.extraction.skills import (
    SkillExtractor,
    SKILL_ALIASES,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


class JobDescriptionMatcher:

    def __init__(
        self,
        skill_extractor: SkillExtractor = None,
    ):
        self.skill_extractor = (
            skill_extractor
            or SkillExtractor()
        )

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:

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
        Normalize a skill using the canonical skill vocabulary.

        This follows the same normalization strategy used by
        ATSAnalyzer.
        """

        if not value:
            return ""

        normalized = re.sub(
            r"\s+",
            " ",
            value.lower().strip(),
        )

        normalized = SKILL_ALIASES.get(
            normalized,
            normalized,
        )

        return re.sub(
            r"\s+",
            " ",
            normalized.lower().strip(),
        )

    # =========================================================
    # SAFE TERM MATCHING
    # =========================================================

    @staticmethod
    def _contains_term(
        text: str,
        term: str,
    ) -> bool:
        """
        Perform safe case-insensitive phrase matching.

        Prevents accidental substring matches such as:
            R -> random
        """

        if not text or not term:
            return False

        text = str(text)
        term = str(term).strip()

        if not term:
            return False

        pattern = (
            r"(?<!\w)"
            + re.escape(term)
            + r"(?!\w)"
        )

        return bool(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
        )

    # =========================================================
    # RESUME EVIDENCE CORPUS
    # =========================================================

    def _build_evidence_corpus(
        self,
        resume: CanonicalResume,
    ) -> List[Tuple[str, str]]:
        """
        Build a complete structured resume evidence corpus.

        Every item is:

            (source, text)

        Example:

            (
                "experience[0].technologies[2]",
                "PyTorch"
            )
        """

        corpus: List[
            Tuple[str, str]
        ] = []

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

            prefix = (
                f"experience[{index}]"
            )

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

            # Experience bullets
            for bullet_index, bullet in enumerate(
                experience.bullet_points or []
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

            # Experience technologies
            for tech_index, technology in enumerate(
                experience.technologies_mentioned or []
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

            prefix = (
                f"education[{index}]"
            )

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
                education.details or []
            ):

                detail_text = (
                    self._normalize_text(
                        detail
                    )
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

            prefix = (
                f"project[{index}]"
            )

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

            for tech_index, technology in enumerate(
                project.technologies or []
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

            prefix = (
                f"certification[{index}]"
            )

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
        # Awards
        # -----------------------------------------------------

        for index, award in enumerate(
            resume.awards
        ):

            award_text = (
                self._normalize_text(
                    award
                )
            )

            if award_text:

                corpus.append(
                    (
                        f"award[{index}]",
                        award_text,
                    )
                )

        # -----------------------------------------------------
        # Volunteer Experience
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
            (source, text)
            for source, text in corpus
            if text
        ]

    # =========================================================
    # SKILL EVIDENCE
    # =========================================================

    def _find_skill_evidence(
        self,
        skill: str,
        corpus: List[Tuple[str, str]],
    ) -> List[str]:
        """
        Find human-readable evidence for a JD skill.
        """

        evidence = []

        for source, text in corpus:

            if self._contains_term(
                text,
                skill,
            ):

                evidence.append(
                    f"{source}: {text}"
                )

        return evidence

    # =========================================================
    # MATCHING
    # =========================================================

    def match(
        self,
        resume: CanonicalResume,
        jd_text: str,
    ) -> Dict[str, Any]:
        """
        Calculate:

        1. JD skill extraction
        2. Resume skill matching
        3. Explainable evidence
        4. Missing skills
        5. TF-IDF semantic similarity
        6. Combined JD similarity score
        """

        # =====================================================
        # VALIDATE JD
        # =====================================================

        if (
            not jd_text
            or len(jd_text.strip()) < 20
        ):

            return {
                "similarity_score": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "matched_skill_evidence": {},
                "missing_skill_evidence": {},
                "total_jd_skills_found": 0,
                "skill_overlap_percentage": 0.0,
                "semantic_similarity": 0.0,
                "key_highlights": (
                    "Please provide a valid Job Description "
                    "of at least 20 characters."
                ),
            }

        # =====================================================
        # 1. EXTRACT JD SKILLS
        # =====================================================

        jd_skills = (
            self.skill_extractor
            .extract_from_text(
                jd_text
            )
        )

        # Remove duplicate canonical skills while
        # preserving original JD ordering.

        unique_jd_skills = []
        seen = set()

        for skill in jd_skills:

            normalized = (
                self._normalize_key(
                    skill
                )
            )

            if (
                normalized
                and normalized not in seen
            ):

                seen.add(normalized)

                unique_jd_skills.append(
                    skill
                )

        jd_skills = unique_jd_skills

        # =====================================================
        # 2. BUILD COMPLETE RESUME EVIDENCE
        # =====================================================

        evidence = (
            self._build_evidence_corpus(
                resume
            )
        )

        all_resume_text = " ".join(
            text
            for _, text in evidence
        )

        # =====================================================
        # 3. MATCH JD SKILLS
        # =====================================================

        matched_skills = []
        missing_skills = []

        matched_skill_evidence = {}
        missing_skill_evidence = {}

        for skill in jd_skills:

            normalized_skill = (
                self._normalize_key(
                    skill
                )
            )

            skill_evidence = (
                self._find_skill_evidence(
                    normalized_skill,
                    evidence,
                )
            )

            if skill_evidence:

                matched_skills.append(
                    skill
                )

                matched_skill_evidence[
                    skill
                ] = skill_evidence

            else:

                missing_skills.append(
                    skill
                )

                missing_skill_evidence[
                    skill
                ] = []

        # =====================================================
        # 4. SKILL OVERLAP
        # =====================================================

        if jd_skills:

            skill_overlap_ratio = (
                len(matched_skills)
                / len(jd_skills)
            )

        else:

            skill_overlap_ratio = 0.5

        skill_overlap_percentage = (
            skill_overlap_ratio * 100.0
        )

        # =====================================================
        # 5. TF-IDF SEMANTIC SIMILARITY
        # =====================================================

        tfidf_similarity = 0.0

        if (
            TfidfVectorizer
            and cosine_similarity
            and all_resume_text.strip()
        ):

            try:

                tfidf = TfidfVectorizer(
                    stop_words="english",
                    max_features=500,
                )

                tfidf_matrix = (
                    tfidf.fit_transform(
                        [
                            all_resume_text,
                            jd_text,
                        ]
                    )
                )

                tfidf_similarity = float(
                    cosine_similarity(
                        tfidf_matrix[0:1],
                        tfidf_matrix[1:2],
                    )[0][0]
                )

            except Exception:

                tfidf_similarity = 0.0

        # =====================================================
        # 6. COMBINED SCORE
        # =====================================================

        # Preserve the existing product weighting:
        #
        # Skill overlap = 60%
        # TF-IDF semantic similarity = 40%
        #
        # Unlike the previous implementation, the TF-IDF
        # component is now kept on a normal 0-100 scale.

        sim_score = (
            skill_overlap_ratio * 60.0
        ) + (
            tfidf_similarity * 40.0
        )

        sim_score = round(
            min(
                98.0,
                max(
                    0.0,
                    sim_score
                ),
            ),
            1,
        )

        # =====================================================
        # 7. RETURN
        # =====================================================

        return {
            "similarity_score": sim_score,

            "skill_overlap_percentage": round(
                skill_overlap_percentage,
                1,
            ),

            "tfidf_similarity_percentage": round(
                tfidf_similarity * 100.0,
                1,
            ),

            "matched_skills": matched_skills,

            "missing_skills": missing_skills,

            "matched_skill_evidence": (
                matched_skill_evidence
            ),

            "missing_skill_evidence": (
                missing_skill_evidence
            ),

            "total_jd_skills_found": len(
                jd_skills
            ),

            "matching_summary": {
                "matched_count": len(
                    matched_skills
                ),
                "missing_count": len(
                    missing_skills
                ),
                "total_count": len(
                    jd_skills
                ),
            },
        }