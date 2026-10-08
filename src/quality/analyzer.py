"""Document extraction quality analysis."""

from collections import Counter
from typing import List

from src.document_models import DocumentBlock

from src.quality.diagnostics import (
    DocumentQualityReport,
    PageQuality,
    QualityIssue,
)

from src.quality.text_quality import (
    character_statistics,
    detect_mojibake,
    text_quality_score,
)


class DocumentQualityAnalyzer:
    """Analyze the reliability of extracted document content."""

    MIN_PAGE_TEXT = 30
    LOW_TEXT_PAGE = 80

    def analyze(
        self,
        blocks: List[DocumentBlock],
    ) -> DocumentQualityReport:
        """Build a quality report from canonical document blocks."""

        report = DocumentQualityReport()

        if not blocks:
            report.overall_confidence = 0.0

            report.add_error(
                QualityIssue(
                    severity="HIGH",
                    category="empty_document",
                    message=(
                        "No document blocks were extracted."
                    ),
                    suggestion=(
                        "Check the parser or use OCR "
                        "for image-based documents."
                    ),
                )
            )

            return report

        # ---------------------------------------------------------
        # BASIC DOCUMENT STATISTICS
        # ---------------------------------------------------------

        report.block_count = len(blocks)

        pages = sorted(
            {
                block.page
                for block in blocks
                if block.page is not None
            }
        )

        report.page_count = len(pages)

        report.extraction_method = (
            blocks[0].extraction_method
            or "unknown"
        )

        total_text = "\n".join(
            block.text
            for block in blocks
            if block.text
        )

        report.total_text_length = len(
            total_text.strip()
        )

        # ---------------------------------------------------------
        # PAGE ANALYSIS
        # ---------------------------------------------------------

        for page_number in pages:

            page_blocks = [
                block
                for block in blocks
                if block.page == page_number
            ]

            page_text = "\n".join(
                block.text
                for block in page_blocks
                if block.text
            ).strip()

            text_length = len(page_text)

            quality = text_quality_score(
                page_text
            )

            page = PageQuality(
                page=page_number,
                text_length=text_length,
                block_count=len(page_blocks),
                has_text=bool(page_text),
                confidence=quality,
            )

            # -----------------------------------------------------
            # Empty page
            # -----------------------------------------------------

            if text_length == 0:

                page.likely_empty = True
                page.likely_scanned = True

                page.issues.append(
                    QualityIssue(
                        severity="HIGH",
                        category="empty_page",
                        message=(
                            f"Page {page_number} "
                            "contains no extracted text."
                        ),
                        suggestion=(
                            "Consider OCR or inspect "
                            "the original page."
                        ),
                    )
                )

            # -----------------------------------------------------
            # Very low text page
            # -----------------------------------------------------

            elif text_length < self.MIN_PAGE_TEXT:

                page.likely_empty = True

                page.issues.append(
                    QualityIssue(
                        severity="MEDIUM",
                        category="low_text",
                        message=(
                            f"Page {page_number} contains "
                            f"only {text_length} characters."
                        ),
                        suggestion=(
                            "Check whether the page contains "
                            "images, graphics, tables, or scanned text."
                        ),
                    )
                )

            elif text_length < self.LOW_TEXT_PAGE:

                page.issues.append(
                    QualityIssue(
                        severity="LOW",
                        category="low_text_density",
                        message=(
                            f"Page {page_number} has relatively "
                            "little extracted text."
                        ),
                        suggestion=(
                            "Inspect the page if important "
                            "content may be graphical."
                        ),
                    )
                )

            # -----------------------------------------------------
            # Encoding quality
            # -----------------------------------------------------

            mojibake = detect_mojibake(
                page_text
            )

            if mojibake:

                report.has_encoding_issues = True

                page.issues.append(
                    QualityIssue(
                        severity="MEDIUM",
                        category="encoding",
                        message=(
                            f"Possible encoding corruption detected "
                            f"on page {page_number}."
                        ),
                        suggestion=(
                            "Review text decoding and PDF extraction."
                        ),
                        evidence={
                            "markers": mojibake,
                        },
                    )
                )

            report.pages.append(page)

        # ---------------------------------------------------------
        # OCR DETECTION
        # ---------------------------------------------------------

        scanned_pages = sum(
            page.likely_scanned
            for page in report.pages
        )

        report.requires_ocr = (
            scanned_pages > 0
            and scanned_pages >= max(
                1,
                report.page_count // 2,
            )
        )

        if report.requires_ocr:

            report.add_warning(
                QualityIssue(
                    severity="HIGH",
                    category="ocr_required",
                    message=(
                        "One or more pages appear to contain "
                        "little or no extractable text."
                    ),
                    suggestion=(
                        "Use OCR or an OCR-capable extraction path."
                    ),
                    evidence={
                        "scanned_pages": scanned_pages,
                        "page_count": report.page_count,
                    },
                )
            )

        # ---------------------------------------------------------
        # DUPLICATE CONTENT
        # ---------------------------------------------------------

        normalized_texts = [
            " ".join(
                block.text.lower().split()
            )
            for block in blocks
            if block.text.strip()
        ]

        counts = Counter(
            normalized_texts
        )

        duplicates = {
            text: count
            for text, count in counts.items()
            if count > 1 and len(text) > 30
        }

        if duplicates:

            report.has_duplicate_content = True

            report.add_warning(
                QualityIssue(
                    severity="LOW",
                    category="duplicate_content",
                    message=(
                        "Repeated extracted text was detected."
                    ),
                    suggestion=(
                        "Check repeated headers, footers, "
                        "or duplicated document content."
                    ),
                    evidence={
                        "duplicate_groups": len(
                            duplicates
                        ),
                    },
                )
            )

        # ---------------------------------------------------------
        # GLOBAL TEXT STATISTICS
        # ---------------------------------------------------------

        stats = character_statistics(
            total_text
        )

        report.metrics.update(
            {
                "character_statistics": stats,
                "scanned_pages": scanned_pages,
                "duplicate_groups": len(
                    duplicates
                ),
            }
        )

        # ---------------------------------------------------------
        # TEXT COVERAGE
        # ---------------------------------------------------------

        pages_with_text = sum(
            page.has_text
            for page in report.pages
        )

        if report.page_count:

            report.text_coverage = (
                pages_with_text
                / report.page_count
            )

        # ---------------------------------------------------------
        # OVERALL CONFIDENCE
        # ---------------------------------------------------------

        page_scores = [
            page.confidence
            for page in report.pages
        ]

        if page_scores:

            report.overall_confidence = (
                sum(page_scores)
                / len(page_scores)
            )

        else:

            report.overall_confidence = 0.0

        # Encoding penalty
        if report.has_encoding_issues:

            report.overall_confidence = max(
                0.0,
                report.overall_confidence - 0.15,
            )

        # OCR penalty
        if report.requires_ocr:

            report.overall_confidence = max(
                0.0,
                report.overall_confidence - 0.25,
            )

        report.overall_confidence = round(
            report.overall_confidence,
            3,
        )

        return report