"""Technical Projects Extractor."""

import re
import uuid
from typing import List, Optional

from src.document_models import DocumentBlock, ProjectEntry
from src.extraction.skills import SkillExtractor


class ProjectExtractor:
    def __init__(
        self,
        skill_extractor: Optional[SkillExtractor] = None,
    ):
        self.skill_extractor = (
            skill_extractor or SkillExtractor()
        )

        self.link_pattern = re.compile(
            r"https?://[^\s\)]+",
            re.IGNORECASE,
        )

    def _is_link_label(self, text: str) -> bool:
        cleaned = text.strip()

        # Remove common bullet characters
        cleaned = re.sub(
            r"^[•·▪◦‣\-*]+\s*",
            "",
            cleaned,
        ).strip()

        return bool(
            re.fullmatch(
                r"(repository|repo|demo|demo\s*&\s*code|"
                r"source\s*code|github|gitlab|live\s*demo)",
                cleaned,
                re.IGNORECASE,
            )
        )    

    def extract(
        self,
        blocks: List[DocumentBlock],
    ) -> List[ProjectEntry]:
        """Parse project entries from the projects section blocks."""

        proj_blocks = [
            b
            for b in blocks
            if b.section == "projects"
            and not b.is_heading
        ]

        if not proj_blocks:
            return []

        # =========================================================
        # SEMICOLON-DELIMITED PROJECTS
        # =========================================================

        full_text = "\n".join(
            b.text for b in proj_blocks
        ).strip()

        lines = [
            line.strip()
            for line in full_text.split("\n")
            if line.strip()
        ]

        if (
            len(lines) <= 3
            and any(";" in line for line in lines)
        ):
            semicolon_projects: List[ProjectEntry] = []

            for line in lines:
                parts = [
                    part.strip().rstrip(". ")
                    for part in line.split(";")
                    if part.strip()
                ]

                for part in parts:
                    if len(part) < 3:
                        continue

                    semicolon_projects.append(
                        ProjectEntry(
                            id=f"proj_{uuid.uuid4().hex[:6]}",
                            title=part,
                            description=part,
                            technologies=(
                                self.skill_extractor
                                .extract_from_text(part)
                            ),
                            confidence=0.88,
                        )
                    )

            if len(semicolon_projects) >= 2:
                return semicolon_projects

        # =========================================================
        # STANDARD PROJECT FORMAT
        # =========================================================

        projects: List[ProjectEntry] = []
        current_project: Optional[ProjectEntry] = None

        for block in proj_blocks:

            text = block.text.strip()

            block_lines = [
                line.strip()
                for line in text.split("\n")
                if line.strip()
            ]

            if not block_lines:
                continue

            # -----------------------------------------------------
            # IMPORTANT:
            # Only the FIRST line of a block can be a new project.
            #
            # This prevents:
            #
            # AI Resume Analyzer
            # Built a resume analysis system.
            #
            # from becoming TWO projects simply because the
            # DocumentBlock itself is bold.
            # -----------------------------------------------------

            first_line = block_lines[0]
            normalized_first_line = first_line.lstrip("•·-* ").strip()

            # =========================================================
            # URL-ONLY BLOCK
            # =========================================================

            if current_project:
                links = self.link_pattern.findall(text)

                if links and not current_project.link:
                    current_project.link = links[0]
                    continue
            
            # =========================================================
            # LINK-ONLY / LINK-LABEL BLOCK
            # =========================================================

            if self._is_link_label(first_line):

                links = self.link_pattern.findall(text)

                if current_project and links:
                    if not current_project.link:
                        current_project.link = links[0]

                # If the label and URL are in separate blocks,
                # the next block may contain the actual URL.
                if current_project and not links:
                    current_project.description = (
                        current_project.description
                    )

                continue

            # =====================================================
            # DETERMINE WHETHER THIS BLOCK STARTS A NEW PROJECT
            # =====================================================

            is_new_project = (
                (
                    block.is_bold
                    and len(first_line.split()) <= 8
                )
                or (
                    ":" in first_line
                    and len(
                        first_line.split(":")[0].split()
                    ) <= 6
                )
                or current_project is None
            )

            if (
                is_new_project
                and current_project is not None
                and len(current_project.description) > 20
            ):
                projects.append(current_project)
                current_project = None

            # =====================================================
            # CREATE NEW PROJECT
            # =====================================================

            if current_project is None:

                title_candidate = first_line

                if (
                    ":"
                    in first_line
                    and len(
                        first_line.split(":")[0].split()
                    ) <= 6
                ):
                    title_candidate = (
                        first_line.split(":")[0]
                        .strip()
                    )

                current_project = ProjectEntry(
                    id=f"proj_{uuid.uuid4().hex[:6]}",
                    title=title_candidate,
                    description=text,
                    technologies=(
                        self.skill_extractor
                        .extract_from_text(text)
                    ),
                    confidence=0.85,
                )

                links = self.link_pattern.findall(text)

                if links:
                    current_project.link = links[0]

                continue

            # =====================================================
            # APPEND REMAINING BLOCK CONTENT
            # =====================================================

            # The entire block belongs to the current project.
            #
            # Do NOT independently evaluate each line for
            # "is_new_project".
            #
            # This is the key fix for the failing tests.

            if text != current_project.description:
                current_project.description += "\n" + text

            technologies = (
                self.skill_extractor
                .extract_from_text(text)
            )

            for technology in technologies:
                if (
                    technology
                    not in current_project.technologies
                ):
                    current_project.technologies.append(
                        technology
                    )

            links = self.link_pattern.findall(text)

            if links and not current_project.link:
                current_project.link = links[0]

        # =========================================================
        # FINAL PROJECT
        # =========================================================

        if current_project:
            projects.append(current_project)

        return projects