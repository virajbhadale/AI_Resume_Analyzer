"""
Universal Education / Academic Qualification Extractor.

Designed to handle common resume layouts such as:

1. Degree
   Institution
   Date
   GPA

2. Institution
   Degree
   Date
   GPA

3. Degree, Institution (Date), GPA

4. Institution
   Pursuing Degree (Date)
   GPA

5. School qualifications:
   Class X
   ABC Public School
   March 2020
   Percentage - 84.8%

The extractor is intentionally evidence-based and does not require
one fixed resume template.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional, Tuple

from src.document_models import DocumentBlock, EducationEntry


# ---------------------------------------------------------------------------
# Degree / qualification detection
# ---------------------------------------------------------------------------

DEGREE_PATTERN = re.compile(
    r"""
    (?<![A-Za-z])
    (?:
        M\.?\s*Sc
        |B\.?\s*Tech
        |B\.?\s*E
        |B\.?\s*S
        |B\.?\s*Sc
        |B\.?\s*A
        |M\.?\s*Tech
        |M\.?\s*E
        |M\.?\s*S
        |Ph\.?\s*D
        |MBA
        |BBA
        |BCA
        |MCA
        |B\.?\s*Com
        |M\.?\s*Com

        |Bachelor(?:['’]s)?(?:\s+of)?(?:\s+\w+){0,8}
        |Master(?:['’]s)?(?:\s+of)?(?:\s+\w+){0,8}
        |Doctor(?:ate)?(?:\s+of)?(?:\s+\w+){0,8}
        |Associate(?:\s+Degree)?
        |Diploma

        |Class\s+(?:XII|XI|X|12th|10th|12|10)
        |(?:10th|12th)(?:\s+(?:Grade|Standard|Class))?
        |Std\.?\s*(?:10|12|10th|12th)
        |Grade\s+(?:10|12|10th|12th)
        |HSC
        |SSC
        |SSLC
        |CBSE
        |ICSE
        |Senior\s+Secondary
        |Higher\s+Secondary
        |Secondary\s+School
        |Intermediate
        |Matriculation
        |High\s+School
    )
    (?![A-Za-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# GPA / percentage detection
# ---------------------------------------------------------------------------

GPA_PATTERN = re.compile(
    r"""
    (?:
        \b(?:CGPA|GPA|Grade)
        \s*(?:scored|score|obtained|of)?\s*
        [:=\-]?\s*
        (?P<score>\d{1,2}(?:\.\d+)?)
        \s*(?:/\s*(?P<scale>\d{1,2}(?:\.\d+)?))?
    )
    |
    (?:
        \b(?:Percentage|Percent|Marks|Score)
        \s*(?:scored|score|obtained|of)?\s*
        [:=\-]?\s*
        (?P<percentage>\d{1,3}(?:\.\d+)?)\s*%
    )
    |
    (?:
        (?P<bare_percentage>\d{1,3}(?:\.\d+)?)\s*%
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Date detection
# ---------------------------------------------------------------------------

MONTH_PATTERN = (
    r"(?:"
    r"Jan(?:uary)?|"
    r"Feb(?:ruary)?|"
    r"Mar(?:ch)?|"
    r"Apr(?:il)?|"
    r"May|"
    r"Jun(?:e)?|"
    r"Jul(?:y)?|"
    r"Aug(?:ust)?|"
    r"Sep(?:tember)?|"
    r"Oct(?:ober)?|"
    r"Nov(?:ember)?|"
    r"Dec(?:ember)?"
    r")"
)

CURRENT_PATTERN = r"(?:Present|Current|Ongoing|Till\s+Date|Now)"

DATE_RANGE_PATTERN = re.compile(
    rf"""
    (?P<start>
        (?:\d{{1,2}}\s+)?{MONTH_PATTERN}\s+\d{{4}}
        |
        \d{{1,2}}/\d{{4}}
        |
        (?:19|20)\d{{2}}
    )
    \s*
    (?:-|–|—|to)
    \s*
    (?P<end>
        (?:\d{{1,2}}\s+)?{MONTH_PATTERN}\s+\d{{4}}
        |
        \d{{1,2}}/\d{{4}}
        |
        (?:19|20)\d{{2}}
        |
        \d{{2}}
        |
        {CURRENT_PATTERN}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

SINGLE_DATE_PATTERN = re.compile(
    rf"""
    (?:
        (?:\d{{1,2}}\s+)?{MONTH_PATTERN}\s+(?:19|20)\d{{2}}
        |
        \b(?:19|20)\d{{2}}\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")


# ---------------------------------------------------------------------------
# Institution detection
# ---------------------------------------------------------------------------

INSTITUTION_KEYWORDS = (
    "university",
    "institute",
    "institution",
    "college",
    "school",
    "academy",
    "polytechnic",
    "campus",
    "junior college",
    "public school",
    "high school",
    "vidya",
    "vidyalaya",
    "kendra",
    "iit",
    "nit",
    "iiit",
    "bits",
    "stanford",
    "berkeley",
    "harvard",
    "mit",
    "cmu",
    "oxford",
    "cambridge",
    "princeton",
    "yale",
    "board",
    "cbse",
    "icse",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_line(text: str) -> str:
    """
    Remove common resume bullets and normalize whitespace.
    """
    text = text.strip()

    text = re.sub(
        r"^[\s•▪▫–—\-\*►✓✔●○◦o]+\s*",
        "",
        text,
    )

    return _normalise_whitespace(text)


def _is_heading(text: str) -> bool:
    normalized = _normalise_whitespace(text).lower()

    return normalized in {
        "education",
        "academic background",
        "academic qualifications",
        "academic qualification",
        "educational background",
        "education & qualifications",
        "education and qualifications",
        "qualifications",
    }


def _degree_match(text: str) -> Optional[re.Match]:
    return DEGREE_PATTERN.search(text)


def _is_degree_line(text: str) -> bool:
    return _degree_match(text) is not None


def _looks_like_institution(text: str) -> bool:
    """
    Determine whether a line is likely to be an institution.

    This deliberately uses several signals rather than requiring one
    exact institution naming pattern.
    """
    text = _normalise_whitespace(text)

    if not text:
        return False

    if _is_heading(text):
        return False

    # A date/GPA-only line is not an institution.
    if _is_date_only(text) or _is_gpa_only(text):
        return False

    # Degree lines should not be classified as institutions.
    if _is_degree_line(text):
        return False

    lower = text.lower()

    if any(keyword in lower for keyword in INSTITUTION_KEYWORDS):
        return True

    # Common institution naming patterns.
    if re.search(
        r"\b(?:univ|university|inst|institute|college|school|academy|poly)\.?\b",
        lower,
    ):
        return True

    return False


def _is_gpa_only(text: str) -> bool:
    text = _normalise_whitespace(text)

    if not GPA_PATTERN.search(text):
        return False

    # If the line contains meaningful non-GPA text, it may still be
    # an education line.
    cleaned = GPA_PATTERN.sub("", text)
    cleaned = re.sub(r"[\s:=,\-–—()]+", "", cleaned)

    return not cleaned


def _is_date_only(text: str) -> bool:
    text = _normalise_whitespace(text)

    if not text:
        return False

    # Date range.
    if DATE_RANGE_PATTERN.fullmatch(text):
        return True

    # Single month/year.
    if SINGLE_DATE_PATTERN.fullmatch(text):
        return True

    # Present/current by itself.
    if re.fullmatch(CURRENT_PATTERN, text, re.IGNORECASE):
        return True

    # Common "March 2022" / "2022" lines with labels.
    cleaned = re.sub(
        r"^(?:graduated|completed|passed|year|date)\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    if SINGLE_DATE_PATTERN.fullmatch(cleaned):
        return True

    return False


def _normalise_short_year(year: str, reference_start: Optional[str] = None) -> str:
    """
    Convert a short year such as 27 into 2027.

    For resume education dates, 20xx is the overwhelmingly common
    interpretation. If a full start year exists, use its century.
    """
    year = year.strip()

    if len(year) == 4:
        return year

    if len(year) == 2:
        if reference_start and len(reference_start) == 4:
            century = reference_start[:2]
            return century + year

        return "20" + year

    return year


def _extract_dates(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract education dates.

    Examples:
        2020 - 2024
        2020 – 2024
        2025-27
        2021 - Present
        March 2020
        2022
    """
    text = _normalise_whitespace(text)

    match = DATE_RANGE_PATTERN.search(text)

    if match:
        start = match.group("start").strip()
        end = match.group("end").strip()

        # Normalize short year ranges.
        if re.fullmatch(r"\d{2}", end):
            start_year_match = re.search(r"\b(?:19|20)\d{2}\b", start)

            if start_year_match:
                end = _normalise_short_year(
                    end,
                    reference_start=start_year_match.group(0),
                )
            else:
                end = _normalise_short_year(end)

        return start, end

    # "Since Jan 2024" / "From 2024"
    since_match = re.search(
        rf"\b(?:since|from)\s+"
        rf"(?P<start>(?:\d{{1,2}}\s+)?{MONTH_PATTERN}\s+\d{{4}}|(?:19|20)\d{{2}})",
        text,
        re.IGNORECASE,
    )

    if since_match:
        return since_match.group("start").strip(), "Present"

    # Single date.

    single = SINGLE_DATE_PATTERN.search(text)

    if single:
        value = single.group(0).strip()

        # Normalize month + year to year only.
        month_year = re.search(
            rf"\b{MONTH_PATTERN}\s+((?:19|20)\d{{2}})\b",
            value,
            re.IGNORECASE,
        )

        if month_year:
            return None, month_year.group(1)

        # For a single education date, treat it as the end/completion date.
        return None, value

    return None, None


def _extract_gpa(text: str) -> Optional[str]:
    """
    Return the normalized score representation.

    Examples:
        CGPA 8.62/10 -> 8.62/10
        GPA: 3.9/4.0 -> 3.9/4.0
        Percentage - 91.4% -> 91.4%
        84.8% -> 84.8%
    """
    match = GPA_PATTERN.search(text)

    if not match:
        return None

    score = match.group("score")
    scale = match.group("scale")
    percentage = match.group("percentage")
    bare_percentage = match.group("bare_percentage")

    if score:
        if scale:
            return f"{score}/{scale}"

        return score

    if percentage:
        return f"{percentage}%"

    if bare_percentage:
        return f"{bare_percentage}%"

    return None


def _remove_degree_prefix(text: str) -> str:
    """
    Remove only the degree portion from a line while preserving
    institution/date/GPA information.
    """
    match = _degree_match(text)

    if not match:
        return text.strip()

    before = text[:match.start()].strip(" ,;|")
    after = text[match.end():].strip(" ,;|")

    # Preserve words such as "in Computer Science" after short degree
    # abbreviations. The regex generally consumes the academic title,
    # not all specialization text.
    if before and after:
        return f"{before} {after}".strip()

    return after or before


def _institution_from_line(text: str) -> Optional[str]:
    """
    Extract an institution candidate from a line containing an education
    record, without returning dates or GPA labels.
    """
    text = _normalise_whitespace(text)

    if not text:
        return None

    # Remove parenthesized date information.
    cleaned = re.sub(
        rf"\([^)]*(?:19|20)\d{{2}}[^)]*\)",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove GPA/percentage portions.
    cleaned = GPA_PATTERN.sub(" ", cleaned)

    # Remove explicit date ranges and standalone dates.
    cleaned = DATE_RANGE_PATTERN.sub(" ", cleaned)
    cleaned = SINGLE_DATE_PATTERN.sub(" ", cleaned)

    # Remove degree portion.
    cleaned = _remove_degree_prefix(cleaned)

    # Remove common pursuing/status prefixes.
    cleaned = re.sub(
        r"\b(?:pursuing|currently\s+pursuing|expected|ongoing)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove common field labels.
    cleaned = re.sub(
        r"\b(?:education|degree|institution|university|college)\s*[:\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Normalize separators.
    cleaned = re.sub(r"^[,;|:\-–—]+|[,;|:\-–—]+$", "", cleaned)
    cleaned = _normalise_whitespace(cleaned)

    if not cleaned:
        return None

    # If the cleaned text itself strongly resembles an institution,
    # accept it.
    if _looks_like_institution(cleaned):
        return cleaned

    # Handle common inline formats where the institution is surrounded
    # by commas.
    candidates = [
        part.strip()
        for part in re.split(r"\s*[,|;]\s*", cleaned)
        if part.strip()
    ]

    for candidate in candidates:
        if _looks_like_institution(candidate):
            return candidate

    # Avoid returning a degree/specialization as an institution.
    if _is_degree_line(cleaned):
        return None

    return None


def _candidate_institution_from_plain_line(text: str) -> Optional[str]:
    """
    Conservative institution detection for a line separate from the
    degree line.
    """
    text = _normalise_whitespace(text)

    if not text:
        return None

    if _is_heading(text):
        return None

    if _is_degree_line(text):
        return None

    if _is_date_only(text):
        return None

    if _is_gpa_only(text):
        return None

    if _looks_like_institution(text):
        return text

    return None


def _split_block_lines(block: DocumentBlock) -> List[str]:
    """
    Preserve physical line ordering while also handling OCR/PDF blocks
    containing multiple logical lines.
    """
    result: List[str] = []

    for raw_line in block.text.splitlines():
        line = _clean_line(raw_line)

        if not line:
            continue

        # Semicolon is a useful record separator in plain-text resumes.
        if ";" in line and not re.search(r"\d;\d", line):
            parts = [p.strip() for p in line.split(";") if p.strip()]
            result.extend(parts)
        else:
            result.append(line)

    return result


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class EducationExtractor:
    """
    Universal education extractor.

    The algorithm is sequence-aware:

        institution -> degree -> date -> GPA
        degree -> institution -> date -> GPA
        degree, institution, date, GPA

    It uses neighboring lines instead of assuming that all information
    exists on the degree line.
    """

    def __init__(self):
        pass

    def _new_entry(
        self,
        degree: str,
        institution: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        gpa: Optional[str] = None,
        details: Optional[List[str]] = None,
        confidence: float = 0.85,
    ) -> EducationEntry:
        return EducationEntry(
            id=f"edu_{uuid.uuid4().hex[:8]}",
            degree=_normalise_whitespace(degree),
            institution=institution or "Institution",
            start_date=start_date,
            end_date=end_date,
            gpa=gpa,
            details=details or [],
            confidence=confidence,
        )

    def _degree_text(self, line: str) -> str:
        """
        Produce a clean degree title.

        Examples:
            Pursuing BSc Data Science -> BSc Data Science
            B.Tech in Computer Science -> B.Tech in Computer Science
            Senior Secondary (Class XII) -> Senior Secondary (Class XII)
        """
        text = _normalise_whitespace(line)

        # Remove common status prefixes.
        text = re.sub(
            r"^(?:currently\s+)?pursuing\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove trailing institution/date/GPA portions where obvious.
        text = re.sub(
            rf"\([^)]*(?:19|20)\d{{2}}[^)]*\)",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = GPA_PATTERN.sub("", text)
        text = DATE_RANGE_PATTERN.sub("", text)

        # Remove leading/trailing separators.
        text = re.sub(r"^[,;|:\-–—]+|[,;|:\-–—]+$", "", text)

        return _normalise_whitespace(text)

    def _is_new_entry(self, line: str) -> bool:
        return _is_degree_line(line)

    def _attach_supplement(
        self,
        entry: EducationEntry,
        line: str,
    ) -> None:
        """
        Attach institution/date/GPA/detail information to an existing
        education entry.
        """
        line = _normalise_whitespace(line)

        if not line:
            return

        # Institution.
        if entry.institution == "Institution":
            institution = _candidate_institution_from_plain_line(line)

            if institution:
                entry.institution = institution

            else:
                institution = _institution_from_line(line)

                if institution:
                    entry.institution = institution

        # Dates.
        start_date, end_date = _extract_dates(line)

        if start_date and not entry.start_date:
            entry.start_date = start_date

        if end_date and not entry.end_date:
            entry.end_date = end_date

        # GPA / percentage.
        gpa = _extract_gpa(line)

        if gpa and not entry.gpa:
            entry.gpa = gpa

        if line not in entry.details:
            entry.details.append(line)

    def extract(self, blocks: List[DocumentBlock]) -> List[EducationEntry]:
        """
        Extract education records from canonical DocumentBlock objects.
        """

        if not blocks:
            return []

        # ---------------------------------------------------------------
        # 1. Select education blocks
        # ---------------------------------------------------------------

        edu_blocks = [
            block
            for block in blocks
            if block.section == "education"
            and not block.is_heading
            and not _is_heading(block.text)
        ]

        # If section tagging is unavailable, use degree/institution/date/score signals.
        if not edu_blocks:
            edu_blocks = [
                block
                for block in blocks
                if (
                    _degree_match(block.text)
                    or _looks_like_institution(block.text)
                    or _is_date_only(block.text)
                    or _is_gpa_only(block.text)
                )
            ]

        if not edu_blocks:
            return []

        # ---------------------------------------------------------------
        # 2. Flatten into ordered logical lines
        # ---------------------------------------------------------------

        raw_lines: List[str] = []

        for block in sorted(
            edu_blocks,
            key=lambda b: (
                b.page,
                b.reading_order,
                b.bbox.y0,
                b.bbox.x0,
            ),
        ):
            raw_lines.extend(_split_block_lines(block))

        if not raw_lines:
            return []

        # ---------------------------------------------------------------
        # 3. Sequence-aware extraction
        # ---------------------------------------------------------------

        entries: List[EducationEntry] = []

        current_entry: Optional[EducationEntry] = None
        pending_institution: Optional[str] = None

        for line in raw_lines:
            line = _clean_line(line)

            if not line:
                continue

            # Ignore section headings.
            if _is_heading(line):
                continue

            degree_match = _degree_match(line)

            # -----------------------------------------------------------
            # CASE A: New degree / qualification detected
            # -----------------------------------------------------------

            if degree_match:

                # Finish previous entry.
                if current_entry is not None:
                    entries.append(current_entry)

                degree = self._degree_text(line)

                # Extract inline information.
                institution = _institution_from_line(line)

                start_date, end_date = _extract_dates(line)

                gpa = _extract_gpa(line)

                # If an institution was detected before this degree,
                # attach it.
                if not institution and pending_institution:
                    institution = pending_institution

                current_entry = self._new_entry(
                    degree=degree,
                    institution=institution,
                    start_date=start_date,
                    end_date=end_date,
                    gpa=gpa,
                    details=[line],
                    confidence=0.94,
                )

                pending_institution = None

                continue

            # -----------------------------------------------------------
            # CASE B: Institution appears before degree
            # -----------------------------------------------------------

            institution = _candidate_institution_from_plain_line(line)

            if current_entry is None:

                if institution:
                    pending_institution = institution

                    # Do not create an EducationEntry yet.
                    # The next degree line will become the actual record.
                    continue

                # A date/GPA without a degree is retained as pending
                # evidence only when an entry is already available.
                continue

            # -----------------------------------------------------------
            # CASE C: Existing education entry
            # -----------------------------------------------------------

            self._attach_supplement(current_entry, line)

        # ---------------------------------------------------------------
        # 4. Append final entry
        # ---------------------------------------------------------------

        if current_entry is not None:
            entries.append(current_entry)

        # ---------------------------------------------------------------
        # 5. Post-processing / quality normalization
        # ---------------------------------------------------------------

        normalized_entries: List[EducationEntry] = []

        for entry in entries:

            # Recover institution if it was still unresolved.
            if entry.institution == "Institution":
                for detail in entry.details:
                    candidate = _candidate_institution_from_plain_line(detail)

                    if candidate:
                        entry.institution = candidate
                        break

            # Remove accidental placeholder details.
            entry.details = [
                d for d in entry.details
                if d and d != "Education"
            ]

            # A valid education entry should have at least a degree.
            if not entry.degree.strip():
                continue

            # Slight confidence adjustment based on available evidence.
            evidence_count = 0

            if entry.degree:
                evidence_count += 1

            if entry.institution != "Institution":
                evidence_count += 1

            if entry.start_date or entry.end_date:
                evidence_count += 1

            if entry.gpa:
                evidence_count += 1

            if evidence_count >= 4:
                entry.confidence = max(entry.confidence, 0.98)
            elif evidence_count == 3:
                entry.confidence = max(entry.confidence, 0.95)
            elif evidence_count == 2:
                entry.confidence = max(entry.confidence, 0.90)

            normalized_entries.append(entry)

        return normalized_entries


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_education(
    blocks: List[DocumentBlock],
) -> List[EducationEntry]:
    """
    Convenience wrapper for callers that prefer a functional API.
    """
    return EducationExtractor().extract(blocks)


__all__ = [
    "EducationExtractor",
    "extract_education",
]