"""Evidence-based location extraction for resume content.

Detects locations associated with experience/education entries while
avoiding ordinary comma-separated technical or descriptive text.
"""

import re
from typing import List, Optional, Tuple


class LocationExtractor:

    # Explicit remote/work-mode locations.
    REMOTE_PATTERNS = (
        "remote",
        "fully remote",
        "remote -",
        "remote,",
        "hybrid",
        "work from home",
        "wfh",
    )

    # Common country names.
    COUNTRIES = {
        "india",
        "united states",
        "usa",
        "us",
        "united kingdom",
        "uk",
        "canada",
        "australia",
        "germany",
        "france",
        "singapore",
        "ireland",
        "uae",
        "united arab emirates",
    }

    # Common Indian states / territories.
    INDIAN_STATES = {
        "maharashtra",
        "karnataka",
        "delhi",
        "telangana",
        "tamil nadu",
        "gujarat",
        "rajasthan",
        "kerala",
        "west bengal",
        "uttar pradesh",
        "madhya pradesh",
        "punjab",
        "haryana",
        "goa",
        "bihar",
        "odisha",
        "assam",
    }

    # Common US state abbreviations.
    US_STATE_CODES = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE",
        "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
        "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
        "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
        "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
        "WI", "WY",
    }

    # Obvious non-location vocabulary.
    NON_LOCATION_TERMS = {
        "python",
        "sql",
        "java",
        "javascript",
        "typescript",
        "pandas",
        "numpy",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "machine learning",
        "deep learning",
        "data science",
        "data engineering",
        "feature engineering",
        "cloud",
        "docker",
        "kubernetes",
        "git",
        "github",
        "api",
        "fastapi",
        "flask",
        "django",
        "tableau",
        "power bi",
        "statistics",
        "analytics",
    }

    def __init__(self) -> None:
        self.city_state_pattern = re.compile(
            r"^[A-Za-z][A-Za-z .'-]{1,40},\s*"
            r"[A-Za-z][A-Za-z .'-]{1,40}$"
        )

        self.city_country_pattern = re.compile(
            r"^[A-Za-z][A-Za-z .'-]{1,40},\s*"
            r"[A-Za-z][A-Za-z .'-]{1,40}$"
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = text.replace("â€“", "-")
        text = text.replace("â€”", "-")
        text = re.sub(r"\s+", " ", text)

        return text.strip(" \t,;:|-")

    # ============================================================
    # BASIC VALIDATION
    # ============================================================

    def _is_non_location(self, text: str) -> bool:
        normalized = text.lower().strip()

        if normalized in self.NON_LOCATION_TERMS:
            return True

        # Technical comma-separated lists.
        parts = [
            part.strip().lower()
            for part in text.split(",")
            if part.strip()
        ]

        if len(parts) >= 2:
            technical_hits = sum(
                part in self.NON_LOCATION_TERMS
                for part in parts
            )

            if technical_hits >= 1:
                return True

        return False

    def _has_location_keyword(self, text: str) -> bool:
        lower = text.lower()

        # ------------------------------------------------------------
        # Remote / work-mode terms
        # ------------------------------------------------------------

        if any(
            pattern in lower
            for pattern in self.REMOTE_PATTERNS
        ):
            return True

        # ------------------------------------------------------------
        # Country / state names
        #
        # Use word boundaries so short values such as "us"
        # do not match inside ordinary words like "various".
        # ------------------------------------------------------------

        for country in self.COUNTRIES:

            if re.search(
                rf"\b{re.escape(country)}\b",
                lower,
            ):
                return True

        for state in self.INDIAN_STATES:

            if re.search(
                rf"\b{re.escape(state)}\b",
                lower,
            ):
                return True

        return False

    # ============================================================
    # LOCATION DETECTION
    # ============================================================

    def detect(
        self,
        text: str,
    ) -> Optional[Tuple[str, float]]:
        """Detect a location and return (location, confidence)."""

        if not text:
            return None

        clean = self.normalize(text)

        if not clean:
            return None

        # ------------------------------------------------------------
        # Basic sanity checks
        # ------------------------------------------------------------

        if len(clean) > 80:
            return None

        if len(clean.split()) > 8:
            return None

        if self._is_non_location(clean):
            return None

        lower = clean.lower()

        # ------------------------------------------------------------
        # Explicit remote / work-mode locations
        # ------------------------------------------------------------

        if lower in {
            "remote",
            "fully remote",
            "hybrid",
            "wfh",
            "work from home",
        }:
            return clean, 0.96

        # ------------------------------------------------------------
        # Explicit location markers
        #
        # Examples:
        #   Location - Pune, India
        #   Location: Pune, India
        #   Pune, India
        # ------------------------------------------------------------

        explicit_match = re.search(
            r"(?:location|based in|located in|office in)"
            r"\s*[-:]\s*"
            r"([A-Za-z][A-Za-z .'-]{1,40},\s*"
            r"[A-Za-z][A-Za-z .'-]{1,40})",
            clean,
            re.IGNORECASE,
        )

        if explicit_match:
            location = self.normalize(
                explicit_match.group(1)
            )

            if not self._is_non_location(location):
                return location, 0.96

        # ------------------------------------------------------------
        # Explicit country / state evidence
        # ------------------------------------------------------------

        if self._has_location_keyword(clean):

            # Do not classify long prose containing a country/state
            # as a location.
            if len(clean.split()) <= 5:
                return clean, 0.92

        # ------------------------------------------------------------
        # City, State / City, Country
        #
        # Only accept a clean two-part geographic expression.
        # ------------------------------------------------------------

        if clean.count(",") == 1:

            left, right = [
                part.strip()
                for part in clean.split(",", 1)
            ]

            if (
                1 <= len(left.split()) <= 5
                and 1 <= len(right.split()) <= 5
                and left[:1].isupper()
                and right[:1].isupper()
            ):

                # Right side should have strong geographic evidence.
                right_lower = right.lower()

                geographic_right = (
                    right_lower in self.COUNTRIES
                    or right_lower in self.INDIAN_STATES
                    or right.upper() in self.US_STATE_CODES
                )

                if geographic_right:
                    return clean, 0.90

        return None

    # ============================================================
    # BLOCK EXTRACTION
    # ============================================================

    def extract_from_lines(
        self,
        lines: List[str],
    ) -> Optional[Tuple[str, float]]:
        """Find the strongest location candidate from multiple lines."""

        best: Optional[Tuple[str, float]] = None

        for line in lines:
            result = self.detect(line)

            if not result:
                continue

            if best is None or result[1] > best[1]:
                best = result

        return best

    def extract_from_text(
        self,
        text: str,
    ) -> Optional[Tuple[str, float]]:
        """Extract a location from multiline text."""

        if not text:
            return None

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return self.extract_from_lines(lines)
