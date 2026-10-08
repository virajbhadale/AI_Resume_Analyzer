"""Date Parsing and Normalization Module."""

import re
from typing import Optional, Dict, Any


# ============================================================
# MONTHS
# Supports:
# Jan
# Jan.
# January
# January.
# Feb
# Feb.
# February
# ...
# ============================================================

MONTHS = (
    r"(?:"
    r"Jan(?:uary)?\.?|"
    r"Feb(?:ruary)?\.?|"
    r"Mar(?:ch)?\.?|"
    r"Apr(?:il)?\.?|"
    r"May\.?|"
    r"Jun(?:e)?\.?|"
    r"Jul(?:y)?\.?|"
    r"Aug(?:ust)?\.?|"
    r"Sep(?:tember)?\.?|"
    r"Oct(?:ober)?\.?|"
    r"Nov(?:ember)?\.?|"
    r"Dec(?:ember)?\.?"
    r")"
)


# ============================================================
# CURRENT / PRESENT TERMS
# ============================================================

CURRENT_TERMS = (
    r"(?:Present|Current|Till\s*Date|Ongoing|Now|Today)"
)


# ============================================================
# DATE RANGE
#
# Supports:
#
# Dec. 2020 - Present
# Dec 2020 - Present
# December 2020 - Present
#
# Nov. 2019 - Dec2020
# Dec. 2017 - Jun. 2019
# June. 2017 - Dec. 2017
#
# 2021 - 2023
# 06/2020 - 08/2022
# ============================================================

DATE_VALUE = (
    rf"(?:"
    rf"(?:\d{{1,2}}\s*)?{MONTHS}\s*\d{{4}}"
    rf"|"
    rf"\d{{1,2}}/\d{{4}}"
    rf"|"
    rf"\b20\d{{2}}\b"
    rf"|"
    rf"\b19\d{{2}}\b"
    rf")"
)


DATE_RANGE_REGEX = re.compile(
    rf"(?P<start>{DATE_VALUE})"
    rf"\s*(?:-|–|—|to|â€“|â€”)\s*"
    rf"(?P<end>{DATE_VALUE}|{CURRENT_TERMS})",
    re.IGNORECASE,
)


# ============================================================
# SINGLE DATE
#
# Supports:
#
# Dec. 2020
# Dec 2020
# December 2020
# 2021
# ============================================================

SINGLE_DATE_REGEX = re.compile(
    rf"\b("
    rf"(?:(?P<day>\d{{1,2}}\s*)?"
    rf"(?P<month>{MONTHS})\s*"
    rf"(?P<year>\d{{4}}))"
    rf"|"
    rf"(?P<year_only>20\d{{2}}|19\d{{2}})"
    rf")\b",
    re.IGNORECASE,
)


class DateExtractor:

    def __init__(self):
        pass

    # ========================================================
    # DATE RANGE EXTRACTION
    # ========================================================

    def extract_date_range(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:

        """
        Search for a date range such as:

        June 2023 - Present
        Dec. 2017 - Jun. 2019
        Nov. 2019 - Dec2020
        2021 - 2023
        """

        match = DATE_RANGE_REGEX.search(text)

        if match:

            start_str = match.group("start").strip()
            end_str = match.group("end").strip()

            is_current = bool(
                re.search(
                    CURRENT_TERMS,
                    end_str,
                    re.IGNORECASE,
                )
            )

            return {
                "start_date": start_str,
                "end_date": end_str,
                "is_current": is_current,
                "raw": match.group(0),
                "confidence": 0.95,
            }

        # ====================================================
        # SINGLE DATE + PRESENT
        #
        # Examples:
        #
        # Since Jan 2024
        # Since Jan. 2024
        # From January 2024
        # ====================================================

        since_match = re.search(
            rf"(?:Since|From)\s+"
            rf"(?P<start>(?:{MONTHS}\s*)?\d{{4}})",
            text,
            re.IGNORECASE,
        )

        if since_match:

            return {
                "start_date": since_match.group("start").strip(),
                "end_date": "Present",
                "is_current": True,
                "raw": since_match.group(0),
                "confidence": 0.90,
            }

        return None

    # ========================================================
    # DATE SIGNAL
    # ========================================================

    def has_date_signal(
        self,
        text: str,
    ) -> bool:

        """
        Quick check if a block contains date indications.
        """

        return bool(
            DATE_RANGE_REGEX.search(text)
            or SINGLE_DATE_REGEX.search(text)
        )