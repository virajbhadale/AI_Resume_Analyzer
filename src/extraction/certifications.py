"""Certifications and Credentials Extractor."""
import re
import uuid
from typing import List
from src.document_models import DocumentBlock, CertificationEntry
from src.extraction.dates import DateExtractor

ISSUER_KEYWORDS = [
    "AWS", "Amazon Web Services", "Google Cloud", "Google", "GCP", "Microsoft", "Azure",
    "DeepLearning.AI", "Coursera", "edX", "Databricks", "IBM", "Oracle",
    "Stanford", "Harvard", "Meta", "TensorFlow", "Kubernetes", "Linux Foundation"
]

YEAR_REGEX = re.compile(r"\b(20\d{2}|19\d{2})\b")

class CertificationExtractor:
    def __init__(self):
        self.date_extractor = DateExtractor()

    def extract(self, blocks: List[DocumentBlock]) -> List[CertificationEntry]:
        cert_blocks = [
            b for b in blocks
            if b.section == "certifications" and not b.is_heading
        ]

        if not cert_blocks:
            return []

        certifications: List[CertificationEntry] = []

        # Convert blocks into clean logical lines
        lines = []

        for b in cert_blocks:
            for line in b.text.split("\n"):
                clean_l = re.sub(
                    r"^[\sâ€¢â–ªâ–«â€“â€”\-\*â–ºâœ“o\u2022]+\s*",
                    "",
                    line
                ).strip()

                if not clean_l:
                    continue

                # Support multiple certifications on the same line
                # Example:
                # AWS Certified Machine Learning - Specialty; Google Professional Data Engineer
                parts = [p.strip() for p in clean_l.split(";") if p.strip()]

                lines.extend(parts)

        i = 0

        while i < len(lines):
            current = lines[i].strip()

            if len(current) < 3:
                i += 1
                continue

            # ---------------------------------------------------------
            # CASE 1:
            # Certification name on one line
            # Issuer + date on next line
            #
            # Example:
            # AWS Certified Machine Learning - Specialty
            # Amazon Web Services | 2022
            # ---------------------------------------------------------

            next_line = lines[i + 1].strip() if i + 1 < len(lines) else None

            if next_line:
                next_has_issuer = any(
                    re.fullmatch(
                        rf"\s*{re.escape(ik)}\s*",
                        next_line,
                        re.IGNORECASE
                    )
                    for ik in ISSUER_KEYWORDS
                )

                next_has_year = bool(YEAR_REGEX.search(next_line))

                # Treat the next line as issuer/date metadata only when it
                # looks like metadata rather than another certification name.
                next_is_metadata = (
                    next_has_year
                    or next_has_issuer
                    or "|" in next_line
                )

                if next_is_metadata:
                    name_clean = current

                    issuer = None
                    for ik in ISSUER_KEYWORDS:
                        if re.search(
                            rf"\b{re.escape(ik)}\b",
                            next_line,
                            re.IGNORECASE
                        ):
                            issuer = ik
                            break

                    cert_date = None

                    date_res = self.date_extractor.extract_date_range(next_line)

                    if date_res:
                        cert_date = date_res["start_date"]
                    else:
                        years = YEAR_REGEX.findall(next_line)
                        if years:
                            cert_date = years[-1]

                    certifications.append(
                        CertificationEntry(
                            id=f"cert_{uuid.uuid4().hex[:6]}",
                            name=name_clean,
                            issuer=issuer,
                            date=cert_date,
                            confidence=0.90
                        )
                    )

                    i += 2
                    continue

            # ---------------------------------------------------------
            # CASE 2:
            # Single-line certification
            #
            # Example:
            # AWS Certified Machine Learning - Specialty
            # ---------------------------------------------------------

            item_clean = current.rstrip(". ").strip()

            issuer = None
            for ik in ISSUER_KEYWORDS:
                if re.search(
                    rf"\b{re.escape(ik)}\b",
                    item_clean,
                    re.IGNORECASE
                ):
                    issuer = ik
                    break

            cert_date = None

            date_res = self.date_extractor.extract_date_range(item_clean)

            if date_res:
                cert_date = date_res["start_date"]
                raw_date = date_res.get("raw", "")

                name_clean = item_clean.replace(
                    raw_date,
                    ""
                ).strip("() ,â€“-")
            else:
                years = YEAR_REGEX.findall(item_clean)

                if years:
                    cert_date = years[-1]

                    name_clean = re.sub(
                        rf"\(?\s*{re.escape(cert_date)}\s*\)?",
                        "",
                        item_clean
                    ).strip("() ,â€“-")
                else:
                    name_clean = item_clean

            certifications.append(
                CertificationEntry(
                    id=f"cert_{uuid.uuid4().hex[:6]}",
                    name=name_clean,
                    issuer=issuer,
                    date=cert_date,
                    confidence=0.90
                )
            )

            i += 1

        return certifications