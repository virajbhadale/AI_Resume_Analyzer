"""Universal experience segmentation engine.

Supports:

- PDF blocks
- DOCX paragraphs
- TXT resumes
- role/company/date in separate blocks
- role/company/date in one block
- bullet-based resumes
- paragraph-based resumes
- internships
- research experience
- academic experience
- current jobs
- jobs without explicit dates

The engine uses multiple evidence signals instead of one fixed regex.
"""

import re
import uuid
from typing import List, Optional

from src.document_models import (
    DocumentBlock,
    ExperienceEpisode,
)

from src.inference.anchors import AnchorDetector
from src.cleaner import DocumentCleaner
from src.extraction.location import LocationExtractor


class ExperienceSegmenter:

    def __init__(self):
        self.anchor_detector = AnchorDetector()
        self.cleaner = DocumentCleaner()
        self.location_extractor = LocationExtractor()

    @staticmethod
    def _lines(
        block: DocumentBlock,
    ) -> List[str]:
        return [
            line.strip()
            for line in block.text.split("\n")
            if line.strip()
        ]

    @staticmethod
    def _is_bullet(
        line: str,
    ) -> bool:

        return bool(
            re.match(
                r"^[\s•▪▫–—\-*►✓o\u2022\u2023\u25E6\u2043\u2219]+\s*",
                line,
            )
        )

    @staticmethod
    def _is_short_heading_candidate(
        line: str,
    ) -> bool:

        clean = line.strip()

        if not clean:
            return False

        # Headings are normally very short.
        if len(clean.split()) > 5:
            return False

        if len(clean) > 50:
            return False

        # Sentence-like text is not a heading.
        if re.search(r"[.!?]$", clean):
            return False

        return True

    def _append_description_lines(
        self,
        current: Optional[ExperienceEpisode],
        lines: List[str],
    ) -> None:
        """Preserve lines as description content.

        Used when AnchorDetector produces a weak role/company signal
        but the surrounding block is more likely to be description
        content than a new experience.
        """

        if current is None:
            return

        for line in lines:

            cleaned = (
                self.cleaner
                .strip_bullet(line)
                .strip()
            )

            if not cleaned:
                continue

            if (
                current.role != "Role Not Specified"
                and cleaned == current.role
            ):
                continue

            if (
                current.company != "Company Not Specified"
                and cleaned == current.company
            ):
                continue

            if (
                current.start_date
                and current.start_date in cleaned
            ):
                continue

            if cleaned not in current.bullet_points:
                current.bullet_points.append(cleaned)

    def _is_new_role_signal(
        self,
        block: DocumentBlock,
        role: Optional[str],
        current: Optional[ExperienceEpisode],
    ) -> bool:

        if not role:
            return False

        # Bullet text should never create a new job.
        if self._is_bullet(block.text):
            return False

        if current is None:
            return True

        if current.role == role:
            return False

        # Short role-like block after an established episode.
        return (
            self._is_short_heading_candidate(
                block.text
            )
            and (
                block.is_bold
                or current.bullet_points
                or current.start_date
                or len(current.description) > 40
            )
        )

    @staticmethod
    def _commit(
        episodes: List[ExperienceEpisode],
        current: Optional[ExperienceEpisode],
    ) -> None:

        if current is None:
            return

        current.description = "\n".join(
            current.bullet_points
        ).strip()

        if not current.role:
            current.role = "Role Not Specified"

        if not current.company:
            current.company = "Company Not Specified"

        episodes.append(current)

    def segment_experiences(
        self,
        exp_blocks: List[DocumentBlock],
    ) -> List[ExperienceEpisode]:

        content_blocks = []

        for block in exp_blocks:

            if not block.text.strip():
                continue

            # =========================================================
            # SECTION-AWARE MODE
            # =========================================================
            #
            # If section metadata exists, it is authoritative.
            # Experience extraction must never cross into another
            # canonical resume section.
            #
            # Example:
            #
            # experience → process
            # skills → stop
            # certifications → stop
            # writing → stop
            # education → stop
            #
            # =========================================================

            if block.section:

                if block.section.lower() != "experience":
                    continue

                content_blocks.append(block)
                continue

            # =========================================================
            # SECTION-UNKNOWN FALLBACK
            # =========================================================
            #
            # If the caller has not run SectionExtractor yet,
            # preserve the original anchor-based behavior.
            #
            # This keeps the segmenter usable independently.
            # =========================================================

            anchor = self.anchor_detector.analyze_block(
                block.text,
                is_bold=block.is_bold,
            )

            has_experience_signal = (
                anchor.role
                or anchor.company
                or anchor.start_date
                or anchor.location
            )

            if not block.is_heading or has_experience_signal:
                content_blocks.append(block)

        if not content_blocks:
            return []

        episodes: List[ExperienceEpisode] = []

        current: Optional[ExperienceEpisode] = None

        pending_role: Optional[str] = None
        pending_company: Optional[str] = None
        pending_location: Optional[str] = None
        pending_location_confidence: float = 0.0

        pending_date_start: Optional[str] = None
        pending_date_end: Optional[str] = None
        pending_is_current: bool = False

        def start_episode() -> ExperienceEpisode:

            return ExperienceEpisode(
                id=f"exp_{uuid.uuid4().hex[:8]}",
                role=(
                    pending_role
                    or "Role Not Specified"
                ),
                company=(
                    pending_company
                    or "Company Not Specified"
                ),
                location=pending_location,
                start_date=pending_date_start,
                end_date=pending_date_end,
                is_current=pending_is_current,
                confidence=0.85,
                evidence={
                    "role_conf": (
                        0.88
                        if pending_role
                        else 0.0
                    ),
                    "company_conf": (
                        0.92
                        if pending_company
                        else 0.0
                    ),
                    "date_conf": (
                        0.95
                        if pending_date_start
                        else 0.0
                    ),
                    "location_conf": (
                        pending_location_confidence
                    ),
                    "source_block": None,
                    "typography_bold": False,
                },
            )

        def reset_pending() -> None:

            nonlocal pending_role
            nonlocal pending_company
            nonlocal pending_location
            nonlocal pending_location_confidence
            nonlocal pending_date_start
            nonlocal pending_date_end
            nonlocal pending_is_current

            pending_role = None
            pending_company = None
            pending_location = None
            pending_location_confidence = 0.0
            pending_date_start = None
            pending_date_end = None
            pending_is_current = False

        def commit_current() -> None:

            nonlocal current

            if current is None:
                return

            current.description = "\n".join(
                current.bullet_points
            ).strip()

            if (
                current.role == "Role Not Specified"
                and current.company == "Company Not Specified"
                and not current.bullet_points
            ):
                current = None
                return

            episodes.append(current)
            current = None

        # =============================================================
        # MAIN LOOP
        # =============================================================

        for block in content_blocks:

            lines = self._lines(block)

            if not lines:
                continue

            anchor = self.anchor_detector.analyze_block(
                block.text,
                is_bold=block.is_bold,
            )

            anchor.block_id = block.id

            block_role = anchor.role
            block_company = anchor.company
            block_date_start = anchor.start_date
            block_date_end = anchor.end_date
            block_location = anchor.location

            block_location_confidence = (
                0.72
                if block_location
                else 0.0
            )

            # ---------------------------------------------------------
            # Detect missing information line-by-line
            # ---------------------------------------------------------

            for line in lines:

                if not block_role:

                    detected_role = (
                        self.anchor_detector
                        .detect_role_in_line(line)
                    )

                    if detected_role:
                        block_role = detected_role

                if not block_company:

                    detected_company = (
                        self.anchor_detector
                        .detect_company_in_line(line)
                    )

                    if detected_company:
                        block_company = detected_company

                if not block_location:

                    location_result = (
                        self.location_extractor.detect(line)
                    )

                    if location_result:

                        (
                            block_location,
                            block_location_confidence,
                        ) = location_result

            has_role = bool(block_role)
            has_company = bool(block_company)
            has_date = bool(block_date_start)

            print(
                "SEGMENT DEBUG:",
                block.id,
                repr(block.text),
                "role=", repr(block_role),
                "company=", repr(block_company),
                "date=", repr(block_date_start),
                "location=", repr(block_location),
                "bold=", block.is_bold,
                "current=",
                (
                    None
                    if current is None
                    else {
                        "role": current.role,
                        "company": current.company,
                        "bullets": current.bullet_points,
                    }
                ),
            )

            # =========================================================
            # CASE 0 — COMPLETE EXPERIENCE HEADER
            #
            # ROLE
            # COMPANY
            # DATE
            #
            # The description may exist in the same block OR in
            # subsequent blocks, so keep the episode open.
            # =========================================================

            if has_role and has_company and has_date:

                # If an existing episode exists, commit it first.
                if current is not None:
                    commit_current()

                pending_role = block_role
                pending_company = block_company
                pending_date_start = block_date_start
                pending_date_end = block_date_end
                pending_is_current = anchor.is_current

                if block_location:

                    pending_location = block_location
                    pending_location_confidence = (
                        block_location_confidence
                    )

                current = start_episode()

                current.evidence["source_block"] = block.id
                current.evidence["typography_bold"] = block.is_bold

                # -----------------------------------------------------
                # Extract description/bullets from the same block.
                # -----------------------------------------------------

                for line in lines:

                    cleaned = (
                        self.cleaner
                        .strip_bullet(line)
                    )

                    if not cleaned:
                        continue

                    if (
                        block_role
                        and cleaned == block_role
                    ):
                        continue

                    if (
                        block_company
                        and cleaned == block_company
                    ):
                        continue

                    if (
                        block_date_start
                        and block_date_start in cleaned
                    ):
                        continue

                    if (
                        self._is_short_heading_candidate(cleaned)
                        and block.is_bold
                        and len(cleaned.split()) <= 5
                    ):
                        continue

                    if self._is_bullet(line):
                        current.bullet_points.append(cleaned)
                        continue

                    # Preserve normal description sentences even when short.
                    #
                    # Example:
                    #   Designed machine learning infrastructure.
                    #
                    # A complete experience header may contain its
                    # description in the same block. Sentence-like
                    # text should be preserved regardless of length.
                    if re.search(r"[.!?]$", cleaned):
                        current.bullet_points.append(cleaned)
                        continue

                    # Preserve longer non-heading text.
                    if len(cleaned.split()) >= 6:
                        current.bullet_points.append(cleaned)

                # IMPORTANT:
                # Do NOT commit here.
                # The next block may contain the description.
                reset_pending()

                continue

            # =========================================================
            # CASE 0.5 — ROLE + COMPANY WITHOUT DATE
            #
            # Example:
            #
            # AI Engineer Intern
            # Marva AI
            #
            # No explicit date is present.
            #
            # This is a valid date-less experience header.
            #
            # IMPORTANT:
            # This case is ONLY allowed when there is no existing
            # experience. Otherwise a descriptive phrase that happens
            # to look like both a role and company must not create a
            # new experience.
            # =========================================================

            if (
                has_role
                and has_company
                and not has_date
                and current is None
            ):

                pending_role = block_role
                pending_company = block_company

                if block_location:

                    pending_location = block_location
                    pending_location_confidence = (
                        block_location_confidence
                    )

                current = start_episode()

                current.evidence[
                    "source_block"
                ] = block.id

                current.evidence[
                    "typography_bold"
                ] = block.is_bold

                # -----------------------------------------------------
                # Do not allow header metadata to become description.
                # -----------------------------------------------------

                for line in lines:

                    cleaned = (
                        self.cleaner
                        .strip_bullet(line)
                        .strip()
                    )

                    if not cleaned:
                        continue

                    if (
                        block_role
                        and cleaned == block_role
                    ):
                        continue

                    if (
                        block_company
                        and cleaned == block_company
                    ):
                        continue

                    if self._is_bullet(line):

                        current.bullet_points.append(
                            cleaned
                        )

                reset_pending()

                continue


            # =========================================================
            # CASE 1 — ROLE
            # =========================================================

            if has_role:

                # -----------------------------------------------------
                # EXISTING EXPERIENCE
                # -----------------------------------------------------
                #
                # AnchorDetector may classify ordinary descriptive
                # phrases as roles:
                #
                #   Backend Development
                #   Machine Learning
                #   Data Analysis
                #   Model Deployment
                #
                # Once an experience is already established, these
                # weak role signals must NOT automatically create a
                # new experience.
                #
                # A new experience requires stronger evidence.
                # -----------------------------------------------------

                if current is not None:

                    # -------------------------------------------------
                    # Same role → definitely NOT a new experience.
                    #
                    # Treat the block as description/content.
                    # -------------------------------------------------

                    if current.role == block_role:

                        self._append_description_lines(
                            current,
                            lines,
                        )

                        continue

                    # -------------------------------------------------
                    # Determine whether the current experience is
                    # already sufficiently established.
                    # -------------------------------------------------

                    established_current = (
                        current.company
                        != "Company Not Specified"
                        or current.start_date is not None
                        or bool(current.bullet_points)
                    )

                    if established_current:

                        # -------------------------------------------------
                        # NON-BOLD ROLE SIGNAL
                        #
                        # Non-bold role-like text is weak evidence.
                        #
                        # Examples:
                        #
                        #   Backend Development
                        #   Machine Learning
                        #   Data Analysis
                        #   Model Deployment
                        #
                        # Treat it as description.
                        # -------------------------------------------------

                        if not block.is_bold:

                            self._append_description_lines(
                                current,
                                lines,
                            )

                            continue

                        # -------------------------------------------------
                        # BOLD ROLE SIGNAL
                        #
                        # Bold + short + role-like is stronger evidence,
                        # but we still require the block itself to look
                        # like a heading.
                        # -------------------------------------------------

                        strong_new_role_signal = (
                            block.is_bold
                            and self._is_short_heading_candidate(
                                block.text
                            )
                        )

                        if not strong_new_role_signal:

                            self._append_description_lines(
                                current,
                                lines,
                            )

                            continue

                        # -------------------------------------------------
                        # STRONG ROLE + COMPANY
                        #
                        # ROLE + COMPANY is strong evidence of a NEW
                        # experience even when the new experience has
                        # NO DATE.
                        #
                        # Example:
                        #
                        # Existing:
                        #   Data Analyst
                        #   ABC Corp
                        #   Jan 2023 – Jun 2023
                        #
                        # New:
                        #   Machine Learning Intern
                        #   XYZ Labs
                        #
                        # There is no date for XYZ Labs, but the strong
                        # role + company combination is enough to close
                        # the previous experience and start a new one.
                        # -------------------------------------------------

                        if has_company:

                            # Close the previous experience.
                            commit_current()

                            # Preserve the new experience metadata.
                            pending_role = block_role
                            pending_company = block_company

                            if block_location:

                                pending_location = block_location
                                pending_location_confidence = (
                                    block_location_confidence
                                )

                            # -------------------------------------------------
                            # Create the new experience immediately.
                            #
                            # This is required for date-less experiences
                            # whose description follows in later blocks.
                            # -------------------------------------------------

                            current = start_episode()

                            current.evidence[
                                "source_block"
                            ] = block.id

                            current.evidence[
                                "typography_bold"
                            ] = block.is_bold

                            # Metadata has now been transferred into
                            # the current episode.
                            reset_pending()

                            continue

                        # -------------------------------------------------
                        # STRONG ROLE WITHOUT COMPANY
                        #
                        # Role alone is still not sufficient to safely
                        # create a new experience.
                        #
                        # Keep the existing experience closed and allow
                        # the role to remain pending until stronger
                        # evidence (company/date) appears.
                        # -------------------------------------------------

                        commit_current()

                # ---------------------------------------------------------
                # NO CURRENT EXPERIENCE
                #
                # Only here can a role safely become pending role metadata.
                # ---------------------------------------------------------

                pending_role = block_role

                if block_location:

                    pending_location = block_location
                    pending_location_confidence = (
                        block_location_confidence
                    )

                continue

            # =============================================================
            # CASE 2 — COMPANY
            # =============================================================

            if has_company:

                # ---------------------------------------------------------
                # EXISTING EXPERIENCE
                # ---------------------------------------------------------
                #
                # AnchorDetector can incorrectly classify short descriptive
                # phrases as companies.
                #
                # Examples:
                #
                #   Model Deployment
                #   Backend Development
                #   Data Analysis
                #
                # If an experience is already established and this block is
                # non-bold, a company-only signal is weak evidence.
                #
                # Treat it as description instead of modifying the
                # experience metadata.
                # ---------------------------------------------------------

                if current is not None:

                    established_current = (
                        current.role
                        != "Role Not Specified"
                        or current.company
                        != "Company Not Specified"
                        or current.start_date is not None
                        or bool(current.bullet_points)
                    )

                    # -----------------------------------------------------
                    # NON-BOLD COMPANY SIGNAL
                    #
                    # A non-bold company-only block inside an established
                    # experience is ambiguous.
                    #
                    # Do NOT immediately create/modify an experience.
                    #
                    # Preserve it as description.
                    # -----------------------------------------------------

                    if (
                        established_current
                        and not block.is_bold
                        and not block_role
                        and not block_date_start
                    ):

                        self._append_description_lines(
                            current,
                            lines,
                        )

                        continue

                    # -----------------------------------------------------
                    # STRONGER COMPANY SIGNAL
                    #
                    # If the company appears with stronger structural
                    # evidence, preserve the normal company-handling logic.
                    #
                    # Examples:
                    #
                    #   ROLE
                    #   COMPANY
                    #   DATE
                    #
                    # or a bold company heading.
                    # -----------------------------------------------------

                # ---------------------------------------------------------
                # If a role is already pending, keep role + company
                # together as the beginning of a date-less experience.
                # ---------------------------------------------------------

                if pending_role:

                    # A different company means a new experience.
                    if (
                        current is not None
                        and current.company != "Company Not Specified"
                        and current.company != block_company
                    ):
                        commit_current()

                    pending_company = block_company

                    if block_location:
                        pending_location = block_location
                        pending_location_confidence = (
                            block_location_confidence
                        )

                    # -----------------------------------------------------
                    # Create the experience immediately even when there is
                    # no date.
                    # -----------------------------------------------------

                    if current is None:

                        current = start_episode()

                        current.evidence[
                            "source_block"
                        ] = block.id

                        current.evidence[
                            "typography_bold"
                        ] = block.is_bold

                    continue

                # ---------------------------------------------------------
                # COMPANY WITHOUT A ROLE
                #
                # Keep it pending until a role/date is found.
                # ---------------------------------------------------------

                if (
                    current is not None
                    and current.company != "Company Not Specified"
                    and block_company != current.company
                ):
                    commit_current()

                pending_company = block_company

                if block_location:

                    pending_location = block_location
                    pending_location_confidence = (
                        block_location_confidence
                    )

                continue

            # =========================================================
            # CASE 3 — DATE
            # =========================================================

            if has_date:

                # ---------------------------------------------------------
                # If an existing episode already exists and does not yet
                # have dates, attach the date directly to it.
                #
                # This handles:
                #
                # ROLE
                # COMPANY
                # DATE
                #
                # including date-less episodes that were created before
                # the date block was encountered.
                # ---------------------------------------------------------

                if (
                    current is not None
                    and current.start_date is None
                ):

                    current.start_date = block_date_start
                    current.end_date = block_date_end
                    current.is_current = anchor.is_current

                    current.evidence[
                        "date_conf"
                    ] = anchor.date_confidence

                    current.evidence[
                        "source_block"
                    ] = block.id

                    continue

                # ---------------------------------------------------------
                # Store date as pending information.
                # ---------------------------------------------------------

                pending_date_start = block_date_start
                pending_date_end = block_date_end
                pending_is_current = anchor.is_current

                # ---------------------------------------------------------
                # If role/company information is waiting, create the
                # episode now.
                # ---------------------------------------------------------

                if pending_role or pending_company:

                    if current is not None:

                        if (
                            pending_company
                            and current.company
                            != pending_company
                        ):
                            commit_current()

                        elif (
                            pending_role
                            and current.role
                            != pending_role
                        ):
                            commit_current()

                    current = start_episode()

                    current.evidence[
                        "source_block"
                    ] = block.id

                    current.evidence[
                        "typography_bold"
                    ] = block.is_bold

                    reset_pending()

                continue

            # =========================================================
            # CASE 4 — LOCATION
            # =========================================================

            if block_location:

                if current is not None:

                    if not current.location:

                        current.location = block_location

                        current.evidence[
                            "location_conf"
                        ] = block_location_confidence

                elif pending_role or pending_company:

                    pending_location = block_location
                    pending_location_confidence = (
                        block_location_confidence
                    )

                continue

            # =========================================================
            # CASE 5 — DESCRIPTION / BULLETS
            # =========================================================

            if current is None:
                continue

            for line in lines:

                cleaned = (
                    self.cleaner
                    .strip_bullet(line)
                    .strip()
                )

                if not cleaned:
                    continue

                # ---------------------------------------------------------
                # Never store already-detected role/company/date as
                # description text.
                # ---------------------------------------------------------

                if (
                    block_role
                    and cleaned == block_role
                ):
                    continue

                if (
                    block_company
                    and cleaned == block_company
                ):
                    continue

                if (
                    block_date_start
                    and block_date_start in cleaned
                ):
                    continue

                # ---------------------------------------------------------
                # IMPORTANT:
                # _is_short_heading_candidate() alone must NOT decide
                # whether text is a heading.
                #
                # Normal resume descriptions can be short:
                #
                #   Backend Development
                #   Data Analysis
                #   Model Deployment
                #
                # Therefore only skip a short line when it has strong
                # heading evidence:
                #
                #   1. block is bold
                #   2. line is very short
                #   3. line is NOT sentence-like
                # ---------------------------------------------------------

                is_heading_like = (
                    block.is_bold
                    and len(cleaned.split()) <= 5
                    and not re.search(
                        r"[.!?]$",
                        cleaned
                    )
                )

                if is_heading_like:
                    continue

                # ---------------------------------------------------------
                # Preserve BOTH:
                #
                # 1. Explicit bullets
                # 2. Normal paragraph descriptions
                #
                # Do NOT require bullet syntax.
                # ---------------------------------------------------------

                if self._is_bullet(line):
                    current.bullet_points.append(cleaned)
                    continue

                # ---------------------------------------------------------
                # Normal description paragraph.
                #
                # Accept non-bullet text when it is reasonably
                # descriptive.
                # ---------------------------------------------------------

                if len(cleaned.split()) >= 5:
                    current.bullet_points.append(cleaned)
                    continue

                # ---------------------------------------------------------
                # Short descriptive lines are also valid resume content
                # when they are not headings.
                #
                # Example:
                #   Backend Development
                #   Model Deployment
                #   Data Analysis
                #
                # Since these are non-bold in your test, preserve them.
                # ---------------------------------------------------------

                if not block.is_bold:
                    current.bullet_points.append(cleaned)


        # =============================================================
        # FINAL COMMIT
        # =============================================================

        commit_current()

        # =============================================================
        # FINAL CLEANUP
        # =============================================================

        cleaned_episodes: List[
            ExperienceEpisode
        ] = []

        for episode in episodes:

            if (
                episode.role
                == "Role Not Specified"
                and episode.company
                == "Company Not Specified"
                and not episode.bullet_points
            ):
                continue

            episode.description = "\n".join(
                episode.bullet_points
            ).strip()

            # ---------------------------------------------------------
            # Merge only exact duplicates.
            # ---------------------------------------------------------

            if cleaned_episodes:

                previous = cleaned_episodes[-1]

                same_episode = (
                    previous.role
                    == episode.role
                    and previous.company
                    == episode.company
                    and previous.start_date
                    == episode.start_date
                    and previous.end_date
                    == episode.end_date
                )

                if same_episode:

                    for bullet in episode.bullet_points:

                        if bullet not in previous.bullet_points:

                            previous.bullet_points.append(
                                bullet
                            )

                    previous.description = "\n".join(
                        previous.bullet_points
                    ).strip()

                    continue

            cleaned_episodes.append(
                episode
            )

        return cleaned_episodes