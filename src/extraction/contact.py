"""Contact Information Extractor using robust regex patterns and spatial heuristics."""
import re
from typing import List, Optional, Tuple
from src.document_models import DocumentBlock, ContactInfo

class ContactExtractor:
    def __init__(self):
        # Email pattern
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
        )
        # Phone pattern (supports international, parenthesis, hyphens, spaces)
        self.phone_pattern = re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}\b"
        )
        # LinkedIn pattern
        self.linkedin_pattern = re.compile(
            r"(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|profile)\/[A-Za-z0-9_-]+",
            re.IGNORECASE
        )
        # GitHub pattern
        self.github_pattern = re.compile(
            r"(?:https?:\/\/)?(?:www\.)?github\.com\/[A-Za-z0-9_-]+",
            re.IGNORECASE
        )
        # Portfolio pattern
        self.portfolio_pattern = re.compile(
            r"(?:https?:\/\/)?(?:www\.)?[A-Za-z0-9_-]+\.(?:io|me|dev|app|tech|page|site)(?:\/[A-Za-z0-9_.-]*)?",
            re.IGNORECASE
        )
        # Location keywords
        self.location_pattern = re.compile(
            r"([A-Za-z\s]+,\s*(?:[A-Z]{2}|[A-Za-z]+))"
        )

    def extract_name(self, header_blocks: List[DocumentBlock]) -> Tuple[Optional[str], float]:
        """Detect a candidate name using typography, adjacency and header context.

        Designer resumes frequently render a person's first and last name on
        separate PDF lines. The extractor therefore considers adjacent name-like
        blocks and can safely combine them when typography and spacing agree.
        """

        if not header_blocks:
            return None, 0.0

        # Work only with the earliest blocks in the reconstructed header. This
        # prevents a large-font section heading later in the document from being
        # mistaken for the candidate name.
        candidates = []
        for block in header_blocks[:10]:
            text = block.text.strip().replace("\n", " ")
            if not self._is_name_candidate(text):
                continue
            candidates.append(block)

        if not candidates:
            return None, 0.0

        # Prefer the strongest visual candidate, but first try to combine
        # adjacent first/last-name blocks (e.g. "Viraj" + "Bhadale").
        for idx, first in enumerate(candidates[:6]):
            if first.font_size < 14:
                continue

            parts = [first.text.strip()]
            last = first

            for second in candidates[idx + 1: idx + 3]:
                vertical_gap = second.bbox.y0 - last.bbox.y1
                same_typography = abs(second.font_size - first.font_size) <= 2.5
                nearby = 0 <= vertical_gap <= 28

                if nearby and same_typography:
                    parts.append(second.text.strip())
                    last = second
                else:
                    break

            combined = " ".join(parts).strip()
            if 1 <= len(combined.split()) <= 4:
                return combined, 0.97 if len(parts) > 1 else 0.95

        # Fallback: strongest single name-like block.
        best = max(candidates, key=lambda b: (b.font_size, -b.reading_order))
        return best.text.strip().replace("\n", " "), 0.92

    @staticmethod
    def _is_name_candidate(text: str) -> bool:
        if not text or len(text) < 2 or len(text) > 50:
            return False
        if "@" in text or "http" in text.lower() or "www." in text.lower():
            return False
        if re.search(r"\d", text):
            return False

        words = text.split()
        if not 1 <= len(words) <= 4:
            return False

        # Avoid obvious resume labels/job titles.
        blocked = {
            "resume", "cv", "curriculum vitae", "contact", "about me",
            "skills", "education", "experience", "projects", "certifications",
            "data scientist", "data analyst", "software engineer",
            "machine learning engineer", "ai engineer", "generative ai engineer",
            "ml engineer", "mlops engineer", "student",
        }
        if text.lower() in blocked:
            return False

        # Names generally consist of alphabetic tokens, allowing common
        # apostrophes/hyphens in surnames.
        return all(re.fullmatch(r"[A-Za-z][A-Za-z'’-]*", w) for w in words)

    def extract(self, blocks: List[DocumentBlock]) -> ContactInfo:
        """Extract all contact elements from document blocks."""
        # Contact is usually concentrated in the first 10-15 blocks or blocks tagged 'contact'
        contact_blocks = [b for b in blocks if b.section in ("contact", None) or b.reading_order < 12]
        all_text = "\n".join([b.text for b in contact_blocks])

        contact = ContactInfo()

        # Name
        name, name_conf = self.extract_name(contact_blocks)
        contact.name = name
        if name:
            contact.confidence["name"] = name_conf

        # Email
        emails = self.email_pattern.findall(all_text)
        if emails:
            contact.email = emails[0].strip()
            contact.confidence["email"] = 0.99

        # Phone
        phones = self.phone_pattern.findall(all_text)
        # Filter false positives: string length between 8 and 16 digits
        for p in phones:
            digits = re.sub(r"\D", "", p)
            if 9 <= len(digits) <= 15:
                contact.phone = p.strip()
                contact.confidence["phone"] = 0.95
                break

        # LinkedIn
        li = self.linkedin_pattern.findall(all_text)
        if li:
            contact.linkedin = li[0].strip()
            contact.confidence["linkedin"] = 0.98

        # GitHub
        gh = self.github_pattern.findall(all_text)
        if gh:
            contact.github = gh[0].strip()
            contact.confidence["github"] = 0.98

        # Portfolio
        pf = self.portfolio_pattern.findall(all_text)
        for p in pf:
            if "linkedin" not in p.lower() and "github" not in p.lower():
                contact.portfolio = p.strip()
                contact.confidence["portfolio"] = 0.90
                break

        # Location heuristic: City, Country or City, State
        for line in all_text.split("\n"):
            line_str = line.strip()
            if "@" in line_str or "http" in line_str or len(line_str) > 45:
                continue
            loc_match = self.location_pattern.search(line_str)
            if loc_match and len(line_str.split()) <= 4:
                contact.location = loc_match.group(1).strip()
                contact.confidence["location"] = 0.85
                break

        return contact
