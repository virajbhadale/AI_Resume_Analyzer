"""
Universal resume section detection.

Responsibilities:
- Detect canonical resume section headings.
- Support synonyms and common resume variations.
- Avoid classifying normal sentences as headings.
- Handle typography/layout signals.
- Support unknown/extended sections safely.
"""

import os
import re
import yaml
from typing import List, Dict, Optional, Tuple

from src.document_models import DocumentBlock
from src.preprocessing.unicode_normalizer import normalize_unicode

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


class SectionExtractor:

    def __init__(self, config_path: str = "config/sections.yaml"):
        self.sections_config = self._load_config(config_path)

        # Flatten all configured patterns for fast lookup.
        self.pattern_map = self._build_pattern_map()

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def _load_config(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            return data if isinstance(data, dict) else {}

        except (OSError, yaml.YAMLError):
            return {}

    def _build_pattern_map(self) -> Dict[str, Tuple[str, str]]:
        """
        Creates:

            normalized pattern
                ->
            (canonical section, section key)
        """

        mapping = {}

        for section_key, section_data in self.sections_config.items():

            if not isinstance(section_data, dict):
                continue

            canonical = section_data.get(
                "canonical",
                section_key
            )

            patterns = section_data.get("patterns", [])

            for pattern in patterns:

                if not isinstance(pattern, str):
                    continue

                normalized = self._normalize_heading(pattern)

                if normalized:
                    mapping[normalized] = (
                        canonical,
                        section_key
                    )

        return mapping

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_heading(text: str) -> str:
        """
        Normalize a potential heading.
        """

        if not text:
            return ""

        text = normalize_unicode(text)
        text = text.strip()

        text = text.replace("|", " ")
        text = text.replace("&", "and")

        # Remove trailing punctuation and decorative heading rules.
        #
        # Resume section headings frequently use visual separators such as:
        #     Experience ______________________
        #     Skills ---------------------------
        #     Education .......................
        #
        # These characters are formatting, not part of the heading text.
        text = re.sub(
            r"[\s:;,\-|_.]+$",
            "",
            text,
        )

        # Collapse whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.lower().strip()

    # ============================================================
    # HEADING STRUCTURE CHECKS
    # ============================================================

    @staticmethod
    def _looks_like_sentence(text: str) -> bool:
        """
        Reject normal prose sentences.

        This is one of the most important safeguards.
        """

        text = text.strip()

        if not text:
            return True

        words = text.split()

        # Long lines are usually content rather than headings.
        if len(words) > 7:
            return True

        # Sentence punctuation.
        if re.search(r"[.!?]$", text):
            return True

        # Common sentence structures.
        sentence_starters = {
            "i",
            "my",
            "we",
            "our",
            "the",
            "this",
            "that",
            "these",
            "those",
            "he",
            "she",
            "they",
            "it",
            "worked",
            "developed",
            "created",
            "built",
            "managed",
            "responsible",
            "experienced",
            "experience",
            "proficient",
            "passionate",
            "seeking",
            "looking",
        }

        first_word = re.sub(
            r"[^a-zA-Z]",
            "",
            words[0]
        ).lower()

        if first_word in sentence_starters:
            return True

        return False

    @staticmethod
    def _has_heading_structure(
        text: str,
        is_bold: bool,
        font_size: float
    ) -> bool:
        """
        Determines whether typography supports heading interpretation.
        """

        text = text.strip()

        # Strong visual evidence.
        if is_bold:
            return True

        if font_size >= 12:
            return True

        # Short ALL CAPS lines are common resume headings.
        if (
            len(text) >= 3
            and text.upper() == text
            and any(c.isalpha() for c in text)
        ):
            return True

        return False

    # ============================================================
    # EXACT MATCH
    # ============================================================

    def _exact_match(
        self,
        normalized_text: str
    ) -> Optional[Tuple[str, float]]:

        if normalized_text in self.pattern_map:

            canonical, _ = self.pattern_map[
                normalized_text
            ]

            return canonical, 1.0

        return None

    # ============================================================
    # FUZZY MATCH
    # ============================================================

    def _fuzzy_match(
        self,
        normalized_text: str,
        is_bold: bool,
        font_size: float
    ) -> Optional[Tuple[str, float]]:

        if fuzz is None:
            return None

        best_section = None
        best_score = 0.0

        for pattern, (
            canonical,
            _
        ) in self.pattern_map.items():

            # Only fuzzy-match short heading-like text.
            pattern_words = pattern.split()
            text_words = normalized_text.split()

            if len(text_words) > 6:
                continue

            if len(text_words) > len(pattern_words) + 2:
                continue

            ratio = fuzz.ratio(
                normalized_text,
                pattern
            ) / 100.0

            token_ratio = fuzz.token_sort_ratio(
                normalized_text,
                pattern
            ) / 100.0

            score = max(
                ratio,
                token_ratio
            )

            if score > best_score:
                best_score = score
                best_section = canonical

        if not best_section:
            return None

        # Conservative fuzzy threshold.
        #
        # Do NOT allow weak fuzzy matches to become headings.
        threshold = 0.90

        if best_score < threshold:
            return None

        # Typography can strengthen an already strong match.
        bonus = 0.0

        if is_bold:
            bonus += 0.03

        if font_size >= 12:
            bonus += 0.03

        final_score = min(
            1.0,
            best_score + bonus
        )

        return best_section, final_score

    # ============================================================
    # MAIN CLASSIFIER
    # ============================================================

    def classify_heading(
        self,
        text: str,
        is_bold: bool = False,
        font_size: float = 10.0,
    ) -> Optional[Tuple[str, float]]:
        """
        Determine whether a block is a resume section heading.

        Returns:
            (canonical_section, confidence)
        or:
            None

        This method intentionally remains independent of document position.
        Document-level context is handled by
        ``segment_document_by_sections``.
        """

        if not text:
            return None

        raw_text = text.strip()

        if not raw_text:
            return None

        words = raw_text.split()

        # ------------------------------------------------------------
        # Basic length protection
        # ------------------------------------------------------------

        if len(words) > 7:
            return None

        if len(raw_text) < 3:
            return None

        normalized = self._normalize_heading(raw_text)

        if not normalized:
            return None

        # ------------------------------------------------------------
        # Exact configured heading
        # ------------------------------------------------------------

        exact = self._exact_match(normalized)

        if exact:
            canonical, score = exact
            return canonical, score

        # ------------------------------------------------------------
        # Obvious prose rejection
        # ------------------------------------------------------------

        if self._looks_like_sentence(raw_text):
            return None

        # ------------------------------------------------------------
        # Typography / visual heading evidence
        # ------------------------------------------------------------

        heading_structure = self._has_heading_structure(
            raw_text,
            is_bold=is_bold,
            font_size=font_size,
        )

        if not heading_structure:
            return None

        # ------------------------------------------------------------
        # Conservative fuzzy matching
        # ------------------------------------------------------------

        fuzzy_match = self._fuzzy_match(
            normalized,
            is_bold=is_bold,
            font_size=font_size,
        )

        if fuzzy_match:
            return fuzzy_match

        return None

    def _vertical_gap(
        self,
        previous_block: DocumentBlock,
        current_block: DocumentBlock,
    ) -> float:
        """
        Calculate vertical separation between two blocks.

        Positive value means the current block starts below the
        previous block.
        """

        return (
            current_block.bbox.y0
            - previous_block.bbox.y1
        )


    def _same_visual_region(
        self,
        first: DocumentBlock,
        second: DocumentBlock,
    ) -> bool:
        """
        Determine whether two blocks belong to the same visual region.

        This uses layout metadata only and does not infer semantic meaning.
        """

        if first.page != second.page:
            return False

        if first.column_id != second.column_id:
            return False

        return True


    def _heading_context_score(
        self,
        blocks: List[DocumentBlock],
        index: int,
    ) -> float:
        """
        Calculate document-level contextual support for a section heading.

        Signals:
        - visual continuity with surrounding blocks
        - heading isolation
        - reading-order position
        - typography
        - local spacing

        This is supporting evidence only.
        It does not replace lexical section matching.
        """

        if not blocks:
            return 0.0

        block = blocks[index]

        score = 0.0

        # ------------------------------------------------------------
        # Typography
        # ------------------------------------------------------------

        if block.is_bold:
            score += 0.20

        if block.font_size >= 12:
            score += 0.20

        # ------------------------------------------------------------
        # Short heading-like text
        # ------------------------------------------------------------

        word_count = len(
            block.text.strip().split()
        )

        if 1 <= word_count <= 4:
            score += 0.15

        # ------------------------------------------------------------
        # Previous block
        # ------------------------------------------------------------

        if index > 0:

            previous = blocks[index - 1]

            if self._same_visual_region(
                previous,
                block,
            ):
                score += 0.10

            gap = self._vertical_gap(
                previous,
                block,
            )

            if gap > 4:
                score += 0.10

        # ------------------------------------------------------------
        # Next block
        # ------------------------------------------------------------

        if index + 1 < len(blocks):

            following = blocks[index + 1]

            if self._same_visual_region(
                block,
                following,
            ):
                score += 0.10

            gap = self._vertical_gap(
                block,
                following,
            )

            if gap >= 0:
                score += 0.05

        # ------------------------------------------------------------
        # Layout metadata
        # ------------------------------------------------------------

        if block.reading_order >= 0:
            score += 0.05

        return min(
            1.0,
            score,
        )


    def _is_contextually_supported_heading(
        self,
        blocks: List[DocumentBlock],
        index: int,
    ) -> bool:
        """
        Determine whether a non-exact section candidate has enough
        contextual support to be treated as a heading.
        """

        score = self._heading_context_score(
            blocks,
            index,
        )

        return score >= 0.35

    # ============================================================
    # DOCUMENT SEGMENTATION
    # ============================================================

    def segment_document_by_sections(
        self,
        blocks: List[DocumentBlock],
    ) -> List[DocumentBlock]:
        """
        Assign each document block to its active section.

        Section detection is performed in document order and uses:
            - lexical section evidence
            - typography
            - layout metadata
            - local context
            - neighboring blocks

        Existing public behavior is preserved:
            blocks are mutated and returned.
        """

        current_section = "contact"

        for index, block in enumerate(blocks):

            heading_result = self.classify_heading(
                block.text,
                is_bold=block.is_bold,
                font_size=block.font_size,
            )

            if heading_result:

                section_name, confidence = (
                    heading_result
                )

                # ----------------------------------------------------
                # Exact configured headings are authoritative.
                # ----------------------------------------------------

                normalized = self._normalize_heading(
                    block.text
                )

                exact_match = self._exact_match(
                    normalized
                )

                if exact_match:

                    block.is_heading = True
                    block.semantic_label = (
                        "SECTION_HEADING"
                    )
                    block.section = section_name
                    block.confidence = confidence

                    current_section = section_name

                    continue

                # ----------------------------------------------------
                # Fuzzy candidates require document context.
                # ----------------------------------------------------

                if self._is_contextually_supported_heading(
                    blocks,
                    index,
                ):

                    block.is_heading = True
                    block.semantic_label = (
                        "SECTION_HEADING"
                    )
                    block.section = section_name
                    block.confidence = confidence

                    current_section = section_name

                    continue

            # --------------------------------------------------------
            # Normal content block
            # --------------------------------------------------------

            block.is_heading = False
            block.section = current_section

            if block.semantic_label is None:
                block.semantic_label = "CONTENT"

        return blocks