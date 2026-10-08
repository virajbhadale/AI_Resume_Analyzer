"""Canonical Resume Builder.

Assembles already-extracted resume components into a CanonicalResume.

This module does not perform document parsing or entity extraction.
It also provides a small normalization layer so simple string-based
project/certification inputs can be converted into the canonical
structured models.
"""

import ntpath
from typing import Any, Dict, List, Optional, Union

from src.document_models import (
    CanonicalResume,
    ContactInfo,
    ExperienceEpisode,
    EducationEntry,
    ProjectEntry,
    CertificationEntry,
)


class ResumeBuilder:
    """Builds the canonical structured resume representation."""

    @staticmethod
    def _normalize_projects(
        projects: Optional[List[Union[str, ProjectEntry]]],
    ) -> List[ProjectEntry]:
        """Convert project strings into ProjectEntry objects.

        Existing ProjectEntry objects are preserved unchanged.
        """

        normalized: List[ProjectEntry] = []

        for index, project in enumerate(projects or [], start=1):

            if isinstance(project, ProjectEntry):
                normalized.append(project)

            elif isinstance(project, str):
                title = project.strip()

                if not title:
                    continue

                normalized.append(
                    ProjectEntry(
                        id=f"project-{index}",
                        title=title,
                    )
                )

            else:
                raise TypeError(
                    "Each project must be either a ProjectEntry "
                    "or a string."
                )

        return normalized

    @staticmethod
    def _normalize_certifications(
        certifications: Optional[
            List[Union[str, CertificationEntry]]
        ],
    ) -> List[CertificationEntry]:
        """Convert certification strings into CertificationEntry objects.

        Existing CertificationEntry objects are preserved unchanged.
        """

        normalized: List[CertificationEntry] = []

        for index, certification in enumerate(
            certifications or [],
            start=1,
        ):

            if isinstance(certification, CertificationEntry):
                normalized.append(certification)

            elif isinstance(certification, str):
                name = certification.strip()

                if not name:
                    continue

                normalized.append(
                    CertificationEntry(
                        id=f"certification-{index}",
                        name=name,
                    )
                )

            else:
                raise TypeError(
                    "Each certification must be either a "
                    "CertificationEntry or a string."
                )

        return normalized

    def build(
        self,
        *,
        contact: Optional[ContactInfo] = None,
        current_title: str = "Professional",
        domain_category: str = "General",
        summary: str = "",
        skills: Optional[List[str]] = None,
        experience: Optional[List[ExperienceEpisode]] = None,
        education: Optional[List[EducationEntry]] = None,
        projects: Optional[
            List[Union[str, ProjectEntry]]
        ] = None,
        certifications: Optional[
            List[Union[str, CertificationEntry]]
        ] = None,
        publications: Optional[List[str]] = None,
        awards: Optional[List[str]] = None,
        achievements: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        volunteer_experience: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
    ) -> CanonicalResume:
        """Assemble extracted components into a CanonicalResume."""

        final_metadata = dict(metadata or {})

        if source_file:
            final_metadata.setdefault(
                "source_file",
                ntpath.basename(source_file),
            )

        normalized_projects = self._normalize_projects(
            projects
        )

        normalized_certifications = (
            self._normalize_certifications(
                certifications
            )
        )

        return CanonicalResume(
            contact=contact or ContactInfo(),

            current_title=(
                current_title.strip()
                if current_title and current_title.strip()
                else "Professional"
            ),

            domain_category=(
                domain_category.strip()
                if domain_category and domain_category.strip()
                else "General"
            ),

            summary=summary.strip() if summary else "",

            skills=list(skills or []),

            experience=list(experience or []),

            education=list(education or []),

            projects=normalized_projects,

            certifications=normalized_certifications,

            publications=list(publications or []),

            awards=list(awards or []),

            achievements=list(achievements or []),

            languages=list(languages or []),

            interests=list(interests or []),

            volunteer_experience=list(
                volunteer_experience or []
            ),

            references=list(references or []),

            metadata=final_metadata,
        )