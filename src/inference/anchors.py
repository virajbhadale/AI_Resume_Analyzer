"""Evidence-based anchor detection for resume experience entries.

The detector intentionally uses multiple weak signals instead of relying on a
fixed list of job titles. This makes it useful for technical, research,
academic and internship resumes while reducing false positives from bullets.
"""

import re
from typing import Optional

from pydantic import BaseModel

from src.extraction.dates import DateExtractor


# Broad title vocabulary.
# This is a SIGNAL, not a whitelist.
ROLE_KEYWORDS = [
    "engineer",
    "scientist",
    "developer",
    "analyst",
    "intern",
    "architect",
    "specialist",
    "lead",
    "manager",
    "consultant",
    "researcher",
    "fellow",
    "associate",
    "director",
    "programmer",
    "administrator",
    "officer",
    "designer",
    "technician",
    "trainee",
    "apprentice",
    "coordinator",
    "professor",
    "lecturer",
    "assistant",
    "instructor",
    "student",
    "internship",
    "graduate",
    "devops",
    "sre",
    "qa",
    "tester",
    "practitioner",
    "owner",
    "master",
    "bi developer",
    "business analyst",
    "data engineer",
    "data analyst",
    "ml engineer",
    "ai engineer",
    "machine learning engineer",
    "software engineer",
    "full stack",
    "full-stack",
    "frontend",
    "backend",
    "devops engineer",
    "cloud engineer",
    "cloud architect",
    "solutions architect",
    "technical writer",
    "content writer",
    "product manager",
    "project manager",
    "scrum master",
]

# Terms that commonly form the second half of a role title.
#
# These should NOT automatically be interpreted as company names
# when they appear after a hyphen.
#
# Examples:
#   Senior Consultant - Data Science
#   Software Engineer - Web Developer Intern
#   Machine Learning Engineer - Artificial Intelligence
#
# In these cases the entire line represents a ROLE.
ROLE_CONTINUATION_TERMS = {
    "data science",
    "data scientist",
    "machine learning",
    "deep learning",
    "software engineering",
    "software engineer",
    "web developer",
    "web development",
    "artificial intelligence",
    "ai",
    "ml",
    "analytics",
    "engineering",
    "development",
    "research",
    "technology",
}

ACADEMIC_TITLE_PATTERNS = (
    r"^b\.?e\.?\b",
    r"^b\.?tech\b",
    r"^m\.?e\.?\b",
    r"^m\.?tech\b",
    r"^m\.?sc\b",
    r"^m\.?s\b",
    r"^b\.?sc\b",
    r"^bachelor\b",
    r"^master\b",
    r"^ph\.?d\b",
    r"^doctorate\b",
    r"^mba\b",
    r"^mca\b",
    r"^bca\b",
)

# Organization/legal suffixes.
COMPANY_SUFFIXES = re.compile(
    r"\b(?:"
    r"inc\.?|"
    r"llc|"
    r"ltd\.?|"
    r"limited|"
    r"corp\.?|"
    r"corporation|"
    r"pvt\.?\s*ltd\.?|"
    r"private\s+limited|"
    r"co\.?"
    r")\b",
    re.IGNORECASE,
)

LOCATION_PATTERN = re.compile(
    r"\(\s*Location\s*-\s*(?P<location>[^)]+)\)",
    re.IGNORECASE,
)

# Common action words.
# A line beginning with these is normally a bullet/description rather than
# a job title.
ACTION_START = re.compile(
    r"^(?:"
    r"built|"
    r"developed|"
    r"designed|"
    r"implemented|"
    r"created|"
    r"led|"
    r"managed|"
    r"worked|"
    r"organized|"
    r"organizes|"
    r"conceptualized|"
    r"conceptualizes|"
    r"monitored|"
    r"provided|"
    r"diagnosed|"
    r"engineered|"
    r"trained|"
    r"deployed|"
    r"optimized|"
    r"improved|"
    r"analyzed|"
    r"performed|"
    r"automated|"
    r"delivered|"
    r"collaborated|"
    r"supported|"
    r"maintained|"
    r"integrated|"
    r"architected|"
    r"launched|"
    r"reduced|"
    r"increased|"
    r"generated|"
    r"conducted|"
    r"researched|"
    r"owned|"
    r"drove|"
    r"served|"
    r"assisted|"
    r"helped|"
    r"used|"
    r"using"
    r")\b",
    re.IGNORECASE,
)


BULLET_PREFIX = re.compile(
    r"^[\s•▪▫–—\-*►✓o\u2022\u2023\u25E6\u2043\u2219]+"
)


class AnchorCandidate(BaseModel):
    """Evidence extracted from a possible experience header."""

    block_id: str = ""
    text: str

    role: Optional[str] = None
    company: Optional[str] = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False

    location: Optional[str] = None

    role_confidence: float = 0.0
    company_confidence: float = 0.0
    date_confidence: float = 0.0

    is_anchor_head: bool = False

    # Document/layout provenance. These fields are populated when the
    # detector is run against DocumentBlock objects and are optional for
    # the legacy single-block API.

    section: Optional[str] = None

    page: int = 1

    column_id: int = 0

    reading_order: int = 0

    block_type: str = "text"

    source_type: str = "unknown"


class AnchorDetector:
    """Detect role/company/date anchors using multiple evidence signals."""

    def __init__(self):
        self.date_extractor = DateExtractor()

    @staticmethod
    def _clean(line: str) -> str:
        line = line.replace("\u00a0", " ").strip()
        line = BULLET_PREFIX.sub("", line).strip()
        return re.sub(r"\s+", " ", line)

    @staticmethod
    def _is_location_text(text: str) -> bool:
        """
        Detect common resume location patterns.

        Location detection is intentionally performed before company
        detection because strings such as:

            Pune, India
            Mumbai, India
            Bangalore, Karnataka

        can otherwise be incorrectly classified as organizations.
        """

        if not text:
            return False

        value = text.strip()

        if not value:
            return False

        # ---------------------------------------------------------
        # Explicit "(Location - ...)" format
        # ---------------------------------------------------------

        if re.search(
            r"\(\s*Location\s*[-:]\s*[^)]+\)",
            value,
            re.IGNORECASE,
        ):
            return True

        # ---------------------------------------------------------
        # City, Country / City, State / City, Region
        #
        # Examples:
        #   Pune, India
        #   Mumbai, India
        #   Bangalore, Karnataka
        #   New York, USA
        # ---------------------------------------------------------

        if re.match(
            r"^[A-Za-z][A-Za-z .'-]{1,40},\s*[A-Za-z][A-Za-z .'-]{1,40}$",
            value,
        ):
            return True

        # ---------------------------------------------------------
        # Common Indian locations
        # ---------------------------------------------------------

        indian_locations = (
            "Pune",
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Bengaluru",
            "Hyderabad",
            "Chennai",
            "Kolkata",
            "Noida",
            "Gurgaon",
            "Gurugram",
            "Nagpur",
            "Nashik",
            "Ahmedabad",
            "Jaipur",
            "Indore",
            "Chandigarh",
            "Kochi",
            "Coimbatore",
            "Mysore",
            "Mysuru",
        )

        if re.search(
            rf"\b(?:{'|'.join(map(re.escape, indian_locations))})\b",
            value,
            re.IGNORECASE,
        ):
            return True

        return False    

    def detect_location(self, text: str) -> Optional[str]:
        """
        Detect location-like text independently from company detection.

        This method is intentionally checked before company detection.
        """

        if not text:
            return None

        value = self._clean(text)

        if not value:
            return None

        # Explicit location metadata
        explicit = self.detect_location_in_line(value)

        if explicit:
            return explicit

        # Standalone location text
        if self._is_location_text(value):
            return value

        return None

    @staticmethod
    def _looks_like_sentence(line: str) -> bool:
        if not line:
            return True

        if ACTION_START.search(line):
            return True

        # Avoid ordinary prose such as:
        # "I developed machine learning models"
        # "We built data pipelines"
        if re.match(
            r"^(?:i|we|he|she|they)\s+\w+(?:ed|ing|s)?\b",
            line,
            re.IGNORECASE,
        ):
            return True

        # Profile-like sentence.
        if re.match(
            r"^experienced\s+",
            line,
            re.IGNORECASE,
        ):
            return True

        if line.endswith((".", ";")):
            return True

        if len(re.findall(r"[,.;:]", line)) >= 3:
            return True

        return False

    @staticmethod
    def _is_role_continuation(text: str) -> bool:
        """
        Determine whether a short string is more likely to be the
        continuation/domain of a role title than a company.

        Examples:
            Data Science
            Data Scientist
            Web Developer Intern
            Machine Learning
            Artificial Intelligence
            Deep Learning
            Engineering
            Development
        """

        if not text:
            return False

        value = re.sub(
            r"\s+",
            " ",
            text.strip().lower(),
        )

        if value in ROLE_CONTINUATION_TERMS:
            return True

        # Flexible role/domain continuation patterns
        continuation_patterns = [
            r"\bdata\s+science\b",
            r"\bdata\s+scientist\b",
            r"\bmachine\s+learning\b",
            r"\bdeep\s+learning\b",
            r"\bsoftware\s+engineer(?:ing)?\b",
            r"\bweb\s+developer(?:ment)?\b",
            r"\bartificial\s+intelligence\b",
            r"\bengineering\b",
            r"\bdevelopment\b",
            r"\bresearch\b",
            r"\btechnology\b",
            r"\bintern\b",
            r"\binternship\b",
        ]

        return any(
            re.search(pattern, value)
            for pattern in continuation_patterns
        )

    def detect_role_in_line(self, line: str) -> Optional[str]:
        """Detect a likely job/research/technical title."""

        clean = self._clean(line)

        if not clean:
            return None

        if len(clean) > 85:
            return None

        if len(clean.split()) > 7:
            return None

        if self._looks_like_sentence(clean):
            return None

        if self.date_extractor.has_date_signal(clean):
            return None

        lower = clean.lower()

        keyword_hit = any(
            re.search(
                rf"\b{re.escape(keyword)}\b",
                lower,
            )
            for keyword in ROLE_KEYWORDS
        )

        # ---------------------------------------------------------
        # Hyphenated role titles
        #
        # Example:
        #   Senior Consultant - Data Science
        #   Software Engineer - Web Developer Intern
        #   Machine Learning Engineer - Artificial Intelligence
        #
        # If the right side looks like a role/domain continuation,
        # the COMPLETE line is a role.
        # ---------------------------------------------------------

        hyphen_match = re.match(
            r"^(.+?)\s*[-–—]\s*(.+)$",
            clean,
        )

        if hyphen_match:
            left = hyphen_match.group(1).strip()
            right = hyphen_match.group(2).strip()

            left_has_role = any(
                re.search(
                    rf"\b{re.escape(keyword)}\b",
                    left.lower(),
                )
                for keyword in ROLE_KEYWORDS
            )

            right_is_role_continuation = (
                self._is_role_continuation(right)
            )

            if left_has_role and right_is_role_continuation:
                return clean

        # Compound title patterns.
        compound_role = re.search(
            r"\b(?:"
            r"senior|"
            r"junior|"
            r"lead|"
            r"principal|"
            r"staff|"
            r"chief|"
            r"graduate|"
            r"teaching|"
            r"full[- ]stack|"
            r"devops"
            r")\b",
            lower,
        )

        if keyword_hit or compound_role:
            candidate = re.split(
                r"\s*(?:\||@|\bat\b)\s*",
                clean,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]

            candidate = candidate.strip(" -–—:")

            if 1 <= len(candidate.split()) <= 9:
                return candidate

        # IMPORTANT:
        # Do NOT classify arbitrary Title Case text as a role.
        #
        # Example:
        #   "Marva AI"
        #   "John Smith"
        #   "Nexus Analytics"
        #
        # These can easily be companies/names.

        return None

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract an explicit location from an organization line."""

        match = re.search(
            r"\(\s*Location\s*[-:]\s*(.*?)\s*\)",
            text,
            re.IGNORECASE,
        )

        if not match:
            return None

        location = match.group(1).strip()

        return location or None


    def detect_company_in_line(
        self,
        line: str,
    ) -> Optional[str]:
        """Detect a likely organization/company/institution."""

        if BULLET_PREFIX.match(line):
            return None

        # Preserve the original line before _clean() removes
        # bullet characters.
        raw = line.strip()

        # ---------------------------------------------------------
        # A line that explicitly starts with a bullet is normally
        # resume content, not an organization name.
        #
        # Examples:
        #
        #   - Developed machine learning models
        #   - Managed clients
        #   • Built data pipelines
        #   – Improved performance
        #
        # These must never become company candidates.
        # ---------------------------------------------------------

        if re.match(
            r"^[\s•▪▫–—\-*►✓o\u2022\u2023\u25E6\u2043\u2219]",
            raw,
        ):
            return None

        clean = self._clean(line)

        if not clean:
            return None

        # ---------------------------------------------------------
        # LOCATION MUST NEVER BECOME A COMPANY
        #
        # Examples:
        #   Pune, India
        #   Mumbai, India
        #   Bangalore, Karnataka
        #
        # These are metadata, not organizations.
        # ---------------------------------------------------------

        if self.detect_location(line):
            return None

        if len(clean) > 90:
            return None

        if len(clean.split()) > 10:
            return None

        if self.date_extractor.has_date_signal(clean):
            return None

        # ---------------------------------------------------------
        # Extract location before company detection
        # ---------------------------------------------------------

        location = self._extract_location(clean)

        company_candidate = clean

        if location:
            company_candidate = self._remove_location(clean)

        company_candidate = company_candidate.strip(
            " -–—:()"
        )

        # ---------------------------------------------------------
        # Strong organization suffix
        #
        # IMPORTANT:
        # Do NOT reject periods here.
        #
        # Valid company names commonly contain:
        #   Pvt Ltd.
        #   Inc.
        #   Ltd.
        #   Corp.
        #   Co.
        # ---------------------------------------------------------

        if COMPANY_SUFFIXES.search(company_candidate):
            return company_candidate.strip(
                " -–—:()"
            )

        # ---------------------------------------------------------
        # Reject obvious sentence-like text
        # ---------------------------------------------------------

        if self._looks_like_sentence(company_candidate):
            return None

        # ---------------------------------------------------------
        # "Role at Company"
        # "Role @ Company"
        # ---------------------------------------------------------

        match = re.search(
            r"(?:\bat\b|@)\s+(.+)$",
            clean,
            re.IGNORECASE,
        )

        if match and len(match.group(1).split()) <= 8:

            candidate = match.group(1).strip(
                " -–—:()."
            )

            if not self._looks_like_sentence(candidate):
                return candidate

        # ---------------------------------------------------------
        # Short organization-like name without suffix
        #
        # Example:
        #   Marva AI (Location - Pune, India)
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Standalone organization name
        #
        # Examples:
        #   ComplyHappy
        #   ByzWiz
        #   Nexus Analytics
        #   Marva AI
        #   Google
        #   Microsoft
        #   Tata Consultancy Services
        #
        # This is allowed only for short, non-sentence-like
        # lines. Role detection is checked first so titles such
        # as "Data Analyst Intern" are not treated as companies.
        # ---------------------------------------------------------

        if self._looks_like_company_name(company_candidate):

            # Prevent role titles from becoming companies.
            if self.detect_role_in_line(company_candidate):
                return None

            return company_candidate.strip(
                " -–—:()"
            )

        return None

    def _looks_like_company_name(self, text: str) -> bool:
        """Heuristic for short company/institution names without legal suffix."""

        text = text.strip()

        if not text:
            return False

        # ---------------------------------------------------------
        # Academic degrees are not organizations.
        # ---------------------------------------------------------

        if self._is_academic_title(text):
            return False

        # ---------------------------------------------------------
        # Locations are not organizations.
        # ---------------------------------------------------------

        if self._is_location_text(text):
            return False

        # ---------------------------------------------------------
        # Role continuation/domain terms are not organizations.
        # ---------------------------------------------------------

        if self._is_role_continuation(text):
            return False

        # ---------------------------------------------------------
        # Reject PDF line-wrap fragments.
        #
        # Example:
        #
        #     ing, Data Engineering, Docker
        # ---------------------------------------------------------

        if re.match(
            r"^[a-z]{2,5},\s+",
            text,
        ):
            return False

        words = text.split()

        # Keep this deliberately conservative.
        if not 1 <= len(words) <= 5:
            return False

        if len(text) > 50:
            return False

        # ---------------------------------------------------------
        # Reject obvious punctuation-heavy fragments.
        # ---------------------------------------------------------

        if text.startswith((",", ".", ":", ";")):
            return False

        if text.endswith((",", ".", ":", ";")):
            return False

        if text.count(",") >= 2:
            return False

        if text.count(":") >= 1:
            return False

        # A company name should not look like a sentence.
        if self._looks_like_sentence(text):
            return False

        # Reject obvious action/sentence starters.
        sentence_starters = {
            "developed",
            "worked",
            "working",
            "engaged",
            "created",
            "built",
            "designed",
            "implemented",
            "utilized",
            "managed",
            "analyzed",
            "analyze",
            "responsible",
            "assisted",
            "led",
            "conducted",
            "improved",
            "expanded",
            "expanding",
            "optimized",
            "optimizing",
            "provided",
            "performed",
        }

        if words[0].lower() in sentence_starters:
            return False

        # ---------------------------------------------------------
        # Reject generic resume/content headings.
        # ---------------------------------------------------------

        generic_content_phrases = {
            "backend development",
            "frontend development",
            "software development",
            "web development",
            "mobile development",
            "machine learning",
            "deep learning",
            "data analysis",
            "data analytics",
            "data science",
            "artificial intelligence",
            "natural language processing",
            "computer vision",
            "project management",
            "product management",
            "technical skills",
            "key skills",
            "professional skills",
            "responsibilities",
            "key responsibilities",
            "key achievements",
            "professional experience",
            "work experience",
            "research experience",
            "academic experience",
            "selected projects",
            "projects",
            "writing",
            "blog writing",
            "technical writing",
            "content writing",
            "publication",
            "publications",
            "certification",
            "certifications",
            "education",
            "experience",
            "work experience",
            "professional experience",
            "skills",
            "technical skills",
            "projects",
            "project",
            "interests",
            "languages",
            "awards",
            "achievements",
            "volunteering",
            "references",
        }

        if text.lower().strip() in generic_content_phrases:
            return False

        # ---------------------------------------------------------
        # Company names generally contain at least one
        # capitalized token / acronym.
        # ---------------------------------------------------------

        has_name_like_token = any(
            word[:1].isupper() or word.isupper()
            for word in words
        )

        if not has_name_like_token:
            return False

        return True


    @staticmethod
    def _is_academic_title(text: str) -> bool:
        """
        Detect academic degree/title lines so they are not classified
        as companies.

        Examples:
            B.E. IN INFORMATION TECHNOLOGY
            B.Tech Computer Science
            Master of Computer Applications
            Ph.D. in Computer Science
        """

        if not text:
            return False

        value = text.strip()

        return any(
            re.search(
                pattern,
                value,
                re.IGNORECASE,
            )
            for pattern in ACADEMIC_TITLE_PATTERNS
        )

    def _parse_role_company_line(
        self,
        line: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Parse compact experience headers such as:

            Senior AI Engineer - TechVision AI Pvt. Ltd.
            Data Scientist - Analytics Hub Solutions

        The parser is intentionally conservative.

        A hyphen does NOT automatically mean:

            ROLE - COMPANY

        because many legitimate role titles themselves contain hyphens:

            Senior Consultant - Data Science
            Software Engineer - Web Developer Intern
            AI Engineer - Machine Learning

        Therefore the right-hand side must have organization-like evidence.
        """

        clean = self._clean(line)

        if not clean:
            return None, None

        # ---------------------------------------------------------
        # Remove trailing date parentheses.
        #
        # Example:
        #
        # Senior AI Engineer - TechVision AI Pvt. Ltd.
        # (Jan 2024-Present)
        #
        # becomes:
        #
        # Senior AI Engineer - TechVision AI Pvt. Ltd.
        # ---------------------------------------------------------

        without_date = re.sub(
            r"\s*\([^)]*\)\s*$",
            "",
            clean,
        ).strip()

        # ---------------------------------------------------------
        # Look for a dash separating role and company.
        #
        # Supported:
        #   -
        #   –
        #   —
        #   â€“
        #   â€”
        # ---------------------------------------------------------

        match = re.match(
            r"^(.+?)\s+"
            r"(?:-|–|—|â€“|â€”)\s+"
            r"(.+?)$",
            without_date,
        )

        if not match:
            return None, None

        role = match.group(1).strip()
        company = match.group(2).strip()

        if not role or not company:
            return None, None

        # ---------------------------------------------------------
        # The left side MUST actually look like a role.
        #
        # Example:
        #
        # "Senior AI Engineer"
        #
        # passes.
        #
        # "Neosoft Technologies"
        #
        # does not.
        # ---------------------------------------------------------

        if not self.detect_role_in_line(role):
            return None, None

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # Some hyphenated titles look exactly like:
        #
        #     ROLE - DOMAIN
        #
        # Example:
        #
        #     Senior Consultant - Data Science
        #     Software Engineer - Web Developer Intern
        #
        # Do NOT interpret the right-hand side as a company.
        # ---------------------------------------------------------

        if self._is_role_continuation(company):
            return None, None

        # ---------------------------------------------------------
        # The right-hand side needs organization-like evidence.
        #
        # Strong evidence:
        #
        #     Pvt Ltd
        #     Ltd
        #     Inc
        #     LLC
        #
        # OR:
        #
        #     a short company-like name.
        # ---------------------------------------------------------

        company_is_valid = (
            bool(COMPANY_SUFFIXES.search(company))
            or self._looks_like_company_name(company)
        )

        if not company_is_valid:
            return None, None

        return role, company

    def _parse_compound_role_company_line(
        self,
        line: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Parse delimiter-free compound experience headers.

        Supported structural form:

            <compound role> [optional parenthetical metadata] <company>

        Examples:
            Senior Data Scientist & Machine Learning Engineer BLOCBELT
            Senior Data Scientist ... PURELOGICS
            Senior Data Scientist ... NVIDIA

        The parser is intentionally conservative. It does not assume
        that the final token is a company.
        """

        clean = self._clean(line)

        if not clean:
            return None, None

        # Explicit role/company separators are handled by the
        # dedicated parsing logic in analyze_block().
        #
        # Do not let delimiter-free compound parsing compete with:
        #
        #     Machine Learning Engineer @ OpenAI
        #     Machine Learning Engineer at OpenAI
        #
        if re.search(
            r"(?:@|\bat\b)",
            clean,
            re.IGNORECASE,
        ):
            return None, None

        words = clean.split()

        if not 2 <= len(words) <= 16:
            return None, None

        if self._looks_like_sentence(clean):
            return None, None

        # ---------------------------------------------------------
        # Remove parenthetical metadata from boundary analysis.
        #
        # Examples:
        #
        #   Engineer (Generative AI) PURELOGICS
        #   Engineer (Volunteer) NVIDIA
        #
        # Parenthetical content is contextual metadata, not part
        # of the role/company boundary.
        # ---------------------------------------------------------

        structural = re.sub(
            r"\s*\([^)]*\)",
            "",
            clean,
        ).strip()

        if not structural:
            return None, None

        structural_words = structural.split()

        if len(structural_words) < 2:
            return None, None

        candidates = []

        for index in range(1, len(structural_words)):
            role_candidate = " ".join(
                structural_words[:index]
            ).strip()

            company_candidate = " ".join(
                structural_words[index:]
            ).strip()

            if not role_candidate or not company_candidate:
                continue

            # -----------------------------------------------------
            # ROLE EVIDENCE
            # -----------------------------------------------------

            lower_role = role_candidate.lower()

            role_keyword_hit = any(
                re.search(
                    rf"\b{re.escape(keyword)}\b",
                    lower_role,
                )
                for keyword in ROLE_KEYWORDS
            )

            if not role_keyword_hit:
                continue

            # The candidate suffix must not itself look like
            # another role.
            if self.detect_role_in_line(company_candidate):
                continue

            if self._is_role_continuation(company_candidate):
                continue

            # -----------------------------------------------------
            # COMPANY EVIDENCE
            # -----------------------------------------------------

            if self.date_extractor.has_date_signal(
                company_candidate
            ):
                continue

            if self.detect_location(company_candidate):
                continue

            if self._looks_like_sentence(company_candidate):
                continue

            has_suffix = bool(
                COMPANY_SUFFIXES.search(company_candidate)
            )

            looks_like_company = (
                self._looks_like_company_name(
                    company_candidate
                )
            )

            # A delimiter-free single-token suffix is ambiguous.
            # Generic occupational / affiliation words must not be
            # promoted to organizations merely because they are
            # capitalized.
            generic_non_company_terms = {
                "mentor",
                "volunteer",
                "intern",
                "internship",
                "trainee",
                "student",
                "researcher",
                "research",
                "fellow",
                "associate",
                "candidate",
                "member",
            }

            normalized_company = (
                company_candidate.strip().lower()
            )

            if (
                len(company_candidate.split()) == 1
                and normalized_company
                in generic_non_company_terms
                and not has_suffix
            ):
                continue

            if not (has_suffix or looks_like_company):
                continue

            # -----------------------------------------------------
            # Reject company candidates that contain obvious role
            # vocabulary.
            #
            # Example:
            #
            #   Engineer BLOCBELT
            #
            # must not become the company.
            # -----------------------------------------------------

            company_contains_role_keyword = any(
                re.search(
                    rf"\b{re.escape(keyword)}\b",
                    company_candidate.lower(),
                )
                for keyword in ROLE_KEYWORDS
            )

            if company_contains_role_keyword:
                continue

            # -----------------------------------------------------
            # Score the boundary.
            #
            # Strong organization suffix is the strongest signal.
            # A clean single-token proper-name organization is also
            # useful, but must not contain role evidence.
            # -----------------------------------------------------

            score = 0.0

            if has_suffix:
                score += 5.0

            if len(company_candidate.split()) == 1:
                score += 2.0
            else:
                score += 1.0

            # Prefer a role ending in a recognized role term.
            final_role_word = (
                role_candidate.split()[-1].lower()
            )

            if any(
                re.search(
                    rf"\b{re.escape(keyword)}\b",
                    final_role_word,
                )
                for keyword in ROLE_KEYWORDS
            ):
                score += 2.0

            # Penalize punctuation-heavy company candidates.
            if any(
                token in company_candidate
                for token in ("(", ")", ",", ":")
            ):
                score -= 2.0

            candidates.append(
                (
                    score,
                    index,
                    role_candidate,
                    company_candidate,
                )
            )

        if not candidates:
            # -----------------------------------------------------
            # ROLE-ONLY COMPOUND HEADER
            #
            # Some experience headers contain a strong role prefix
            # followed by affiliation/context text, but no sufficiently
            # reliable organization evidence.
            #
            # Example:
            #
            #   AI Data Scientist Kaggle Master Mentor
            #
            # The evidence supports:
            #
            #   role    = AI Data Scientist
            #   company = None
            #
            # Do not promote the remaining affiliation text into a
            # company merely to complete the pair.
            # -----------------------------------------------------

            role_candidates = []

            for index in range(1, len(structural_words)):
                role_candidate = " ".join(
                    structural_words[:index]
                ).strip()

                suffix = " ".join(
                    structural_words[index:]
                ).strip()

                if not role_candidate or not suffix:
                    continue

                lower_role = role_candidate.lower()

                role_keyword_hit = any(
                    re.search(
                        rf"\b{re.escape(keyword)}\b",
                        lower_role,
                    )
                    for keyword in ROLE_KEYWORDS
                )

                if not role_keyword_hit:
                    continue

                # The role should end at an actual role-bearing
                # term rather than at an arbitrary word.
                final_word = (
                    role_candidate.split()[-1].lower()
                )

                final_word_is_role = any(
                    re.search(
                        rf"\b{re.escape(keyword)}\b",
                        final_word,
                    )
                    for keyword in ROLE_KEYWORDS
                )

                if not final_word_is_role:
                    continue

                # Avoid treating a complete sentence as a role.
                if self._looks_like_sentence(role_candidate):
                    continue

                role_candidates.append(
                    (
                        len(role_candidate.split()),
                        role_candidate,
                    )
                )

            if not role_candidates:
                return None, None

            # Prefer the longest strongly role-terminated prefix.
            _, role = max(
                role_candidates,
                key=lambda item: item[0],
            )

            return role, None

        # Highest evidence score wins.
        #
        # For equal scores, prefer the boundary with the shorter
        # company span because organization names generally form a
        # compact suffix in delimiter-free headers.
        best = max(
            candidates,
            key=lambda item: (
                item[0],
                -len(item[3].split()),
            ),
        )

        _, _, role, company = best

        return role, company


    def analyze_block(
        self,
        block_text: str,
        is_bold: bool = False,
    ) -> AnchorCandidate:
        """Analyze a complete document block."""

        candidate = AnchorCandidate(
            text=block_text
        )

        raw_lines = [
            line
            for line in block_text.split("\n")
            if line.strip()
        ]

        lines = [
            self._clean(line)
            for line in raw_lines
            if self._clean(line)
        ]

        # ---------------------------------------------------------
        # LOCATION
        # ---------------------------------------------------------

        for line in lines:
            location = self.detect_location(line)

            if location and not candidate.location:
                candidate.location = location

        # ---------------------------------------------------------
        # DATE
        # ---------------------------------------------------------

        date_result = self.date_extractor.extract_date_range(
            block_text
        )

        if date_result:
            candidate.start_date = date_result["start_date"]
            candidate.end_date = date_result["end_date"]
            candidate.is_current = date_result["is_current"]
            candidate.date_confidence = date_result["confidence"]

        # ---------------------------------------------------------
        # ROLE + COMPANY
        # ---------------------------------------------------------

        for raw_line, line in zip(raw_lines, lines):

            # -----------------------------------------------------
            # COMPACT FORMAT:
            #
            # Senior AI Engineer - TechVision AI Pvt. Ltd.
            # (Jan 2024-Present)
            #
            # Data Scientist - Analytics Hub Solutions
            # (Jun 2022-Dec 2023)
            # -----------------------------------------------------

            # -----------------------------------------------------
            # COMPACT ROLE + COMPANY FORMAT
            #
            # Explicit ROLE - COMPANY structures are handled first.
            # -----------------------------------------------------

            # COMPACT ROLE + COMPANY FORMAT
            parsed_role, parsed_company = self._parse_role_company_line(line)

            if parsed_role and not candidate.role:
                candidate.role = parsed_role
                candidate.role_confidence = 0.96 if is_bold else 0.94

            if parsed_company and not candidate.company:
                candidate.company = parsed_company
                candidate.company_confidence = 0.94


            # NORMAL ROLE DETECTION
            #
            # This MUST run before the compound parser so that complete,
            # legitimate role titles such as:
            #
            #   Data Analyst Intern
            #   Software Engineer Intern
            #   AI Engineer Intern
            #   Senior Consultant - Data Science
            #
            # are preserved exactly.
            role = self.detect_role_in_line(line)

            if role and not candidate.role:
                candidate.role = role
                candidate.role_confidence = 0.95 if is_bold else 0.88


            # COMPOUND ROLE + COMPANY FALLBACK
            #
            # Only attempt compound decomposition when the normal detectors
            # could not already identify the line.
            if not candidate.role and not candidate.company:
                parsed_role, parsed_company = (
                    self._parse_compound_role_company_line(line)
                )

                if parsed_role:
                    candidate.role = parsed_role
                    candidate.role_confidence = 0.96 if is_bold else 0.94

                if parsed_company:
                    candidate.company = parsed_company
                    candidate.company_confidence = 0.94

            # -----------------------------------------------------
            # COMPOUND ROLE + COMPANY HEADER
            #
            # Supports delimiter-free headers where a compound
            # role is followed by an organization name.
            # -----------------------------------------------------

            if not parsed_role and not parsed_company:
                (
                    parsed_role,
                    parsed_company,
                ) = self._parse_compound_role_company_line(line)

            if parsed_role and not candidate.role:
                candidate.role = parsed_role

                candidate.role_confidence = (
                    0.96
                    if is_bold
                    else 0.94
                )

            if parsed_company and not candidate.company:
                candidate.company = parsed_company
                candidate.company_confidence = 0.94

            # -----------------------------------------------------
            # Normal role detection
            # -----------------------------------------------------

            role = self.detect_role_in_line(line)

            if role and not candidate.role:
                candidate.role = role

                candidate.role_confidence = (
                    0.95
                    if is_bold
                    else 0.88
                )

            # -----------------------------------------------------
            # Normal company detection
            # -----------------------------------------------------

            company = self.detect_company_in_line(raw_line)

            if company and not candidate.company:

                # Never allow a detected role to become a company.
                if candidate.role and company == candidate.role:
                    company = None

                # Never allow a role continuation/domain to become
                # a company.
                if company and self._is_role_continuation(company):
                    company = None

                # Never allow location metadata to become a company.
                if company and self.detect_location(line):
                    company = None

            if company and not candidate.company:
                candidate.company = company
                candidate.company_confidence = 0.92

            # -----------------------------------------------------
            # Same-line format:
            #
            # Data Scientist @ Google
            # ML Engineer at Microsoft
            # -----------------------------------------------------

            match = re.match(
                r"(.+?)\s+(?:at|@)\s+(.+)$",
                line,
                re.IGNORECASE,
            )

            if match:
                left, right = match.groups()

                detected_role = self.detect_role_in_line(left)

                if (
                    not candidate.role
                    and detected_role
                ):
                    candidate.role = detected_role
                    candidate.role_confidence = 0.94

                if (
                    detected_role
                    and not candidate.company
                ):
                    candidate.company = right.strip()
        candidate.company_confidence = 0.90

        # ---------------------------------------------------------
        # COMPANY WITHOUT A LEGAL SUFFIX
        #
        # Example:
        # AI Engineer Intern
        # Marva AI
        # 2025 - 2026
        #
        # "Marva AI" must not be classified as a role.
        # ---------------------------------------------------------

        if not candidate.company and (
            candidate.role or candidate.start_date
        ):
            for raw_line, line in zip(raw_lines, lines):

                if BULLET_PREFIX.match(raw_line):
                    continue
                
                if line == candidate.role:
                    continue

                if self.date_extractor.has_date_signal(line):
                    continue

                # -------------------------------------------------
                # Do not use locations as companies.
                # -------------------------------------------------

                if self.detect_location(line):
                    continue

                # -------------------------------------------------
                # Do not use roles as companies.
                # -------------------------------------------------

                if self.detect_role_in_line(line):
                    continue

                # -------------------------------------------------
                # Do not use academic titles as companies.
                # -------------------------------------------------

                if self._is_academic_title(line):
                    continue

                # -------------------------------------------------
                # Do not use role continuation terms as companies.
                # -------------------------------------------------

                if self._is_role_continuation(line):
                    continue

                # -------------------------------------------------
                # Do not use sentence-like content.
                # -------------------------------------------------

                if self._looks_like_sentence(line):
                    continue

                words = line.split()

                if not 1 <= len(words) <= 6:
                    continue

                # -------------------------------------------------
                # IMPORTANT:
                #
                # Only accept a fallback company if the complete
                # company heuristic agrees.
                # -------------------------------------------------

                if not self._looks_like_company_name(line):
                    continue

                candidate.company = line.strip(
                    " -–—:"
                )

                candidate.company_confidence = 0.72
                break

        # ---------------------------------------------------------
        # FINAL ANCHOR DECISION
        # ---------------------------------------------------------
        #
        # Strong experience anchors:
        #
        # 1. ROLE + DATE
        #
        #       DATA SCIENTIST
        #       Nov. 2019 - Dec. 2020
        #
        # 2. ROLE + COMPANY
        #
        #       DATA SCIENTIST
        #       Infosys
        #
        # 3. COMPANY + DATE + LOCATION
        #
        #       Infosys
        #       Pune, India
        #       Dec. 2017 - Jun. 2019
        #
        # We intentionally do NOT consider:
        #
        #       COMPANY + DATE
        #
        # alone as a strong experience anchor because certifications,
        # education and projects can have the same structure.
        # ---------------------------------------------------------

        role_and_date = bool(
            candidate.role
            and candidate.start_date
        )

        role_and_company = bool(
            candidate.role
            and candidate.company
        )

        company_date_location = bool(
            candidate.company
            and candidate.start_date
            and candidate.location
        )

        candidate.is_anchor_head = bool(
            role_and_date
            or role_and_company
            or company_date_location
        )

        return candidate

    def analyze_document_block(
        self,
        block,
    ) -> AnchorCandidate:
        """Analyze one canonical DocumentBlock while preserving provenance."""

        candidate = self.analyze_block(
            block.text,
            is_bold=block.is_bold,
        )

        candidate.block_id = block.id

        candidate.section = block.section

        candidate.page = block.page

        candidate.column_id = block.column_id

        candidate.reading_order = block.reading_order

        candidate.block_type = block.block_type

        candidate.source_type = block.source_type

        return candidate

    def analyze_document_blocks(
        self,
        blocks,
    ) -> list[AnchorCandidate]:
        """Analyze DocumentBlock objects in their supplied document order."""

        return [
            self.analyze_document_block(block)
            for block in blocks
        ]

    def detect_location_in_line(self, line: str) -> Optional[str]:
        """Detect explicit location metadata."""

        clean = self._clean(line)

        match = LOCATION_PATTERN.search(clean)

        if match:
            return match.group("location").strip()

        return None

    @staticmethod
    def _remove_location(text: str) -> str:
        """Remove explicit '(Location - ...)' metadata."""

        return re.sub(
            r"\s*\(\s*Location\s*-\s*[^)]*\)",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
    
    