"""Metadata resolution and normalization for extracted experience episodes."""

import re
from typing import Any, Dict, List

from src.document_models import ExperienceEpisode


class MetadataResolver:
    """Normalize and validate metadata already extracted from an experience."""

    _MISSING_VALUES = {
        "",
        "none",
        "null",
        "unknown",
        "not specified",
        "n/a",
        "na",
    }

    _CURRENT_TERMS = {
        "present",
        "current",
        "ongoing",
        "now",
        "today",
        "till date",
    }

    def __init__(self) -> None:
        pass

    @classmethod
    def _clean_text(cls, value: Any) -> str:
        """Normalize whitespace and return empty string for missing values."""

        if value is None:
            return ""

        text = str(value).strip()

        if not text:
            return ""

        text = re.sub(r"\s+", " ", text).strip()

        if text.lower() in cls._MISSING_VALUES:
            return ""

        return text

    @classmethod
    def _normalize_location(cls, value: Any) -> str:
        """Normalize an already extracted experience location."""

        location = cls._clean_text(value)

        if not location:
            return ""

        # Normalize common dash spacing.
        location = re.sub(
            r"\s*[-–—]\s*",
            " - ",
            location,
        )

        # Collapse accidental repeated punctuation.
        location = re.sub(
            r"\s*,\s*",
            ", ",
            location,
        )

        return location.strip(" ,;-")

    @classmethod
    def _normalize_current_status(
        cls,
        end_date: Any,
        is_current: bool,
    ) -> tuple[str, bool]:
        """
        Normalize end-date/current status.

        Existing explicit current status is preserved.
        Current terms in the end date are converted to 'Present'.
        """

        end = cls._clean_text(end_date)

        if end and end.lower() in cls._CURRENT_TERMS:
            return "Present", True

        if is_current:
            return "Present", True

        return end, False

    def resolve_episode(self, episode):
        """
        Resolve and normalize metadata already attached to an episode.

        This method does not perform candidate generation, company
        inference, location inference, or cross-episode association.
        """

        # --------------------------------------------------------------
        # ROLE
        # --------------------------------------------------------------

        if episode.role is not None:
            episode.role = str(episode.role).strip()

        # --------------------------------------------------------------
        # COMPANY + EMBEDDED LOCATION
        # --------------------------------------------------------------

        company, embedded_location = (
            self._split_embedded_location(
                episode.company
            )
        )

        if company is not None:
            episode.company = company

        # --------------------------------------------------------------
        # LOCATION
        # --------------------------------------------------------------

        explicit_location = self._normalize_location_value(
            episode.location
        )

        embedded_location = self._normalize_location_value(
            embedded_location
        )

        # Prefer an explicitly supplied episode.location.
        #
        # Embedded company location is used only when the episode
        # does not already contain a location.
        if explicit_location:
            episode.location = explicit_location
        elif embedded_location:
            episode.location = embedded_location
        else:
            episode.location = None

        # --------------------------------------------------------------
        # DATES
        # --------------------------------------------------------------

        if episode.start_date is not None:
            episode.start_date = str(
                episode.start_date
            ).strip() or None

        if episode.end_date is not None:
            episode.end_date = str(
                episode.end_date
            ).strip() or None

        # --------------------------------------------------------------
        # CURRENT STATUS
        # --------------------------------------------------------------

        if episode.is_current and not episode.end_date:
            episode.end_date = "Present"

        return episode

    def resolve(
        self,
        episodes: List[ExperienceEpisode],
    ) -> List[ExperienceEpisode]:
        """Resolve metadata for all experience episodes."""

        return [
            self.resolve_episode(episode)
            for episode in episodes
        ]

    # ==================================================================
    # PHASE 18C.6 — METADATA NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize_location_value(value):
        """
        Normalize an explicitly supplied location value.

        This method only removes structural formatting noise.
        It does NOT infer or classify locations.
        """
        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        # Explicit metadata prefix:
        #
        #   Location - Pune, India
        #   location: Pune, India
        #   LOCATION – Pune, India
        #
        value = re.sub(
            r"^\s*location\s*[-:–—]\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = value.strip(" \t\r\n,;:-–—")

        return value or None

    @staticmethod
    def _split_embedded_location(company):
        """
        Separate an explicitly embedded location from a company value.

        Supported structural form:

            Company Name (Location - Pune, India)

        Returns:
            (clean_company, embedded_location)

        No location inference is performed here.
        """
        if company is None:
            return None, None

        company = str(company).strip()

        if not company:
            return None, None

        match = re.search(
            r"\(\s*location\s*[-:–—]\s*(?P<location>[^)]*?)\s*\)",
            company,
            flags=re.IGNORECASE,
        )

        if not match:
            return company, None

        location = match.group("location").strip()

        clean_company = (
            company[:match.start()]
            + company[match.end():]
        ).strip()

        clean_company = re.sub(
            r"\s{2,}",
            " ",
            clean_company,
        ).strip(" \t\r\n,;:-–—")

        return (
            clean_company or None,
            location or None,
        )