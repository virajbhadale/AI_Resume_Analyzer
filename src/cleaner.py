"""Hygiene and cleaning engine for document blocks."""

import re
from typing import List

from src.document_models import DocumentBlock
from src.preprocessing.unicode_normalizer import normalize_unicode


class DocumentCleaner:
    """
    Cleans already-normalized document blocks.

    Responsibilities:
    - Remove empty blocks
    - Remove page-number-only blocks
    - Normalize block text
    - Strip leading bullet symbols when explicitly requested

    Unicode/mojibake repair is delegated to unicode_normalizer.py.
    """

    def __init__(self, strip_page_numbers: bool = True):
        self.strip_page_numbers = strip_page_numbers

        self.bullet_regex = re.compile(
            r"^[\s"
            r"\u2022"
            r"\u2023"
            r"\u25E6"
            r"\u2043"
            r"\u2219"
            r"\u25AA"
            r"\u25AB"
            r"\u25B6"
            r"\u25BA"
            r"\-\*"
            r"]+\s*"
        )

        self.page_num_regex = re.compile(
            r"^(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?$",
            re.IGNORECASE,
        )

    # ============================================================
    # TEXT CLEANING
    # ============================================================

    def clean_text(self, text: str) -> str:
        """
        Normalize text using the centralized Unicode normalizer.
        """

        if not text:
            return ""

        return normalize_unicode(text)

    # ============================================================
    # BULLET HANDLING
    # ============================================================

    def strip_bullet(self, text: str) -> str:
        """
        Strip leading bullet symbols from a line.
        """

        cleaned = self.clean_text(text)

        if not cleaned:
            return ""

        return self.bullet_regex.sub("", cleaned).strip()

    # ============================================================
    # PAGE NUMBER DETECTION
    # ============================================================

    def is_page_number(self, text: str) -> bool:
        """
        Check whether a block contains only a page number.

        Examples:
            1
            2
            Page 1
            Page 2 of 3
            2/3
        """

        cleaned = self.clean_text(text).lower()

        return bool(
            self.page_num_regex.fullmatch(cleaned)
        )

    # ============================================================
    # BLOCK FILTERING
    # ============================================================

    def filter_blocks(
        self,
        blocks: List[DocumentBlock],
    ) -> List[DocumentBlock]:
        """
        Filter empty blocks, page numbers, and obvious artifacts.
        """

        cleaned_blocks: List[DocumentBlock] = []

        for block in blocks:

            norm_text = self.clean_text(block.text)

            # ----------------------------------------------------
            # Empty block
            # ----------------------------------------------------

            if not norm_text:
                continue

            # ----------------------------------------------------
            # Page number
            # ----------------------------------------------------

            if (
                self.strip_page_numbers
                and self.is_page_number(norm_text)
            ):
                continue

            # ----------------------------------------------------
            # Update block
            # ----------------------------------------------------

            block.text = norm_text

            cleaned_blocks.append(block)

        return cleaned_blocks