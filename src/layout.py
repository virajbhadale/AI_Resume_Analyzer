"""
Production-grade layout and reading-order reconstruction engine.

The public API is intentionally stable. Layout reconstruction is based on
page geometry and local regions rather than a fixed two-column template.
"""

from typing import List, Dict, Tuple, Any
from collections import defaultdict

from src.document_models import DocumentBlock


class LayoutEngine:
    """
    Detect page regions and reconstruct deterministic reading order.

    Supported layouts include:
    - single-column documents
    - two-column documents
    - three-column documents
    - asymmetric/sidebar layouts
    - full-width headers and section bands
    - mixed full-width + column layouts
    - multi-page documents

    Important architectural rule:
        layout reconstruction uses geometry only. Resume semantics are
        resolved later by section/evidence/inference layers.
    """

    def __init__(
        self,
        gutter_threshold: float = 18.0,
        line_tolerance: float = 3.5,
        min_column_ratio: float = 0.18,
    ):
        self.gutter_threshold = gutter_threshold
        self.line_tolerance = line_tolerance
        self.min_column_ratio = min_column_ratio

    # ============================================================
    # BASIC GEOMETRY HELPERS
    # ============================================================

    @staticmethod
    def _center_x(block: DocumentBlock) -> float:
        return block.bbox.center_x

    @staticmethod
    def _center_y(block: DocumentBlock) -> float:
        return block.bbox.center_y

    @staticmethod
    def _width(block: DocumentBlock) -> float:
        return max(0.0, block.bbox.width)

    @staticmethod
    def _height(block: DocumentBlock) -> float:
        return max(0.0, block.bbox.height)

    # ============================================================
    # FULL-WIDTH / HEADER DETECTION
    # ============================================================

    def _is_full_width_block(
        self,
        block: DocumentBlock,
        page_width: float,
    ) -> bool:
        """
        Detect blocks spanning most of the page.

        Used as a broad visual signal only.
        """
        width_ratio = self._width(block) / max(page_width, 1.0)
        return width_ratio >= 0.60

    def _is_region_spanning_block(
        self,
        block: DocumentBlock,
        page_width: float,
    ) -> bool:
        """
        Detect a block that should act as a page-flow boundary.

        This threshold is intentionally stricter than the broad
        full-width classification.
        """
        width_ratio = self._width(block) / max(page_width, 1.0)
        return width_ratio >= 0.72

    def _is_header_block(
        self,
        block: DocumentBlock,
        page_width: float,
    ) -> bool:
        """
        Detect likely global header blocks.

        This remains geometry-based and does not infer candidate identity.
        """
        top_limit = max(150.0, page_width * 0.30)

        return (
            block.bbox.y0 <= top_limit
            and self._is_full_width_block(block, page_width)
        )

    # ============================================================
    # COLUMN CANDIDATE DETECTION
    # ============================================================

    def _generate_split_candidates(
        self,
        blocks: List[DocumentBlock],
        page_width: float,
    ) -> List[float]:
        """
        Generate possible vertical split positions.

        Kept for compatibility with the existing layout diagnostics.
        """
        candidates = []

        for block in blocks:

            if block.bbox.x0 > 0:
                candidates.append(block.bbox.x0)

            if block.bbox.x1 < page_width:
                candidates.append(block.bbox.x1)

            candidates.append(block.bbox.center_x)

        candidates.extend(
            page_width * ratio
            for ratio in (
                0.25,
                0.30,
                0.33,
                0.35,
                0.38,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.62,
                0.65,
                0.67,
                0.70,
                0.75,
            )
        )

        return sorted(
            {
                round(x, 2)
                for x in candidates
                if page_width * 0.20 <= x <= page_width * 0.80
            }
        )

    def _score_split(
        self,
        blocks: List[DocumentBlock],
        split_x: float,
        page_width: float,
    ) -> float:
        """
        Score a two-way split.

        Retained as a compatibility helper for existing callers/tests.
        """
        left = []
        right = []
        crossing = []

        for block in blocks:

            if self._width(block) > page_width * 0.65:
                continue

            if block.bbox.x1 <= split_x:
                left.append(block)

            elif block.bbox.x0 >= split_x:
                right.append(block)

            else:
                crossing.append(block)

        total = len(left) + len(right)

        if total < 4:
            return -1.0

        left_ratio = len(left) / total
        right_ratio = len(right) / total

        if (
            left_ratio < self.min_column_ratio
            or right_ratio < self.min_column_ratio
        ):
            return -1.0

        score = min(left_ratio, right_ratio) * 2.0

        score -= (
            len(crossing)
            / max(len(blocks), 1)
        ) * 2.0

        nearby_left = [
            block.bbox.x1
            for block in left
        ]

        nearby_right = [
            block.bbox.x0
            for block in right
        ]

        if nearby_left and nearby_right:

            gutter = (
                min(nearby_right)
                - max(nearby_left)
            )

            if gutter >= self.gutter_threshold:
                score += 1.0

        return score

    # ============================================================
    # UNIVERSAL COLUMN CLUSTERING
    # ============================================================

    def _cluster_columns(
        self,
        blocks: List[DocumentBlock],
        page_width: float,
    ) -> List[List[DocumentBlock]]:
        """
        Cluster non-spanning blocks into horizontal content regions.

        The algorithm does not assume:
            - 50/50 columns
            - exactly two columns
            - a particular resume template

        A significant center-to-center gap is treated as a potential
        visual gutter.

        At most three regions are emitted. This prevents small
        indentation differences from creating arbitrary fragmentation.
        """

        if not blocks:
            return []

        ordered = sorted(
            blocks,
            key=lambda block: (
                self._center_x(block),
                block.bbox.x0,
                block.bbox.y0,
                block.id,
            ),
        )

        if len(ordered) < 2:
            return [ordered]

        centers = [
            self._center_x(block)
            for block in ordered
        ]

        gaps = [
            centers[index + 1] - centers[index]
            for index in range(len(centers) - 1)
        ]

        gap_threshold = max(
            self.gutter_threshold * 2.0,
            page_width * 0.10,
        )

        split_indexes = [
            index
            for index, gap in enumerate(gaps)
            if gap >= gap_threshold
        ]

        if not split_indexes:
            return [
                sorted(
                    blocks,
                    key=lambda block: (
                        block.bbox.y0,
                        block.bbox.x0,
                        block.id,
                    ),
                )
            ]

        groups: List[List[DocumentBlock]] = []

        start = 0

        for split_index in split_indexes:

            groups.append(
                ordered[
                    start:split_index + 1
                ]
            )

            start = split_index + 1

        groups.append(
            ordered[start:]
        )

        # Prevent pathological fragmentation.
        while len(groups) > 3:

            distances = []

            for index in range(
                len(groups) - 1
            ):

                left_center = (
                    sum(
                        self._center_x(block)
                        for block in groups[index]
                    )
                    / len(groups[index])
                )

                right_center = (
                    sum(
                        self._center_x(block)
                        for block in groups[index + 1]
                    )
                    / len(groups[index + 1])
                )

                distances.append(
                    (
                        right_center - left_center,
                        index,
                    )
                )

            _, merge_index = min(
                distances
            )

            merged = (
                groups[merge_index]
                + groups[merge_index + 1]
            )

            groups[
                merge_index:merge_index + 2
            ] = [merged]

        return [
            sorted(
                group,
                key=lambda block: (
                    block.bbox.y0,
                    block.bbox.x0,
                    block.id,
                ),
            )
            for group in groups
        ]

    # ============================================================
    # COLUMN DETECTION PUBLIC API
    # ============================================================

    def detect_columns_on_page(
        self,
        page_blocks: List[DocumentBlock],
        page_width: float = 612.0,
    ) -> Dict[str, Any]:
        """
        Detect the number of horizontal content regions.

        Existing return structure is preserved.

        For two columns:
            split_x represents the approximate center boundary.

        For one/three columns:
            split_x is diagnostic only.
        """

        if not page_blocks:
            return {
                "num_columns": 1,
                "split_x": page_width / 2.0,
                "confidence": 0.95,
            }

        body_blocks = [
            block
            for block in page_blocks
            if not self._is_region_spanning_block(
                block,
                page_width,
            )
        ]

        if len(body_blocks) < 2:
            return {
                "num_columns": 1,
                "split_x": page_width / 2.0,
                "confidence": 0.95,
            }

        groups = self._cluster_columns(
            body_blocks,
            page_width,
        )

        num_columns = max(
            1,
            min(3, len(groups)),
        )

        if num_columns == 1:

            return {
                "num_columns": 1,
                "split_x": page_width / 2.0,
                "confidence": 0.90,
            }

        if num_columns == 2:

            left_center = (
                sum(
                    self._center_x(block)
                    for block in groups[0]
                )
                / len(groups[0])
            )

            right_center = (
                sum(
                    self._center_x(block)
                    for block in groups[1]
                )
                / len(groups[1])
            )

            split_x = (
                left_center
                + right_center
            ) / 2.0

        else:

            middle_left = max(
                block.bbox.x1
                for block in groups[0]
            )

            middle_right = min(
                block.bbox.x0
                for block in groups[1]
            )

            split_x = (
                middle_left
                + middle_right
            ) / 2.0

        confidence = min(
            0.99,
            0.70
            + 0.08 * (num_columns - 1),
        )

        return {
            "num_columns": num_columns,
            "split_x": split_x,
            "confidence": confidence,
        }

    # ============================================================
    # BLOCK SORTING
    # ============================================================

    def _sort_blocks_vertically(
        self,
        blocks: List[DocumentBlock],
    ) -> List[DocumentBlock]:
        """
        Sort blocks top-to-bottom, then left-to-right.
        """

        return sorted(
            blocks,
            key=lambda block: (
                round(
                    block.bbox.y0
                    / self.line_tolerance
                ),
                block.bbox.x0,
                block.id,
            ),
        )

    # ============================================================
    # COLUMN ASSIGNMENT
    # ============================================================

    def _assign_column(
        self,
        block: DocumentBlock,
        split_x: float,
    ) -> int:
        """
        Backward-compatible two-column assignment helper.
        """

        return (
            0
            if block.bbox.center_x <= split_x
            else 1
        )

    # ============================================================
    # REGION ORDERING
    # ============================================================

    def _order_region(
        self,
        blocks: List[DocumentBlock],
        page_width: float,
    ) -> List[Tuple[DocumentBlock, int]]:
        """
        Order one local page region.

        Multiple horizontal regions are read:
            left -> right

        Each region is read:
            top -> bottom
        """

        if not blocks:
            return []

        groups = self._cluster_columns(
            blocks,
            page_width,
        )

        result: List[
            Tuple[DocumentBlock, int]
        ] = []

        for column_id, group in enumerate(groups):

            for block in group:

                result.append(
                    (
                        block,
                        column_id,
                    )
                )

        return result

    # ============================================================
    # PAGE FLOW
    # ============================================================

    def _order_page(
        self,
        page_blocks: List[DocumentBlock],
        page_width: float,
    ) -> List[Tuple[DocumentBlock, int]]:
        """
        Reconstruct page flow while preserving full-width/column
        transitions.

        Example:

            FULL WIDTH HEADER
                    ↓
            LEFT COLUMN | RIGHT COLUMN
                    ↓
            FULL WIDTH SECTION HEADING
                    ↓
            LEFT COLUMN | RIGHT COLUMN

        This is intentionally geometry-only.
        """

        if not page_blocks:
            return []

        spanning = [
            block
            for block in page_blocks
            if self._is_region_spanning_block(
                block,
                page_width,
            )
        ]

        if not spanning:

            return self._order_region(
                page_blocks,
                page_width,
            )

        spanning = sorted(
            spanning,
            key=lambda block: (
                block.bbox.y0,
                block.bbox.x0,
                block.id,
            ),
        )

        spanning_ids = {
            block.id
            for block in spanning
        }

        body = [
            block
            for block in page_blocks
            if block.id not in spanning_ids
        ]

        ordered: List[
            Tuple[DocumentBlock, int]
        ] = []

        previous_y = float("-inf")

        for span in spanning:

            span_y = self._center_y(
                span
            )

            before = [
                block
                for block in body
                if (
                    previous_y
                    < self._center_y(block)
                    < span_y
                )
            ]

            ordered.extend(
                self._order_region(
                    before,
                    page_width,
                )
            )

            ordered.append(
                (
                    span,
                    0,
                )
            )

            previous_y = span_y

        after = [
            block
            for block in body
            if self._center_y(block)
            >= previous_y
        ]

        ordered.extend(
            self._order_region(
                after,
                page_width,
            )
        )

        # Defensive invariant:
        # every source block must appear exactly once.
        seen = {
            block.id
            for block, _ in ordered
        }

        if len(seen) != len(page_blocks):

            missing = [
                block
                for block in page_blocks
                if block.id not in seen
            ]

            missing = self._sort_blocks_vertically(
                missing
            )

            ordered.extend(
                [
                    (
                        block,
                        0,
                    )
                    for block in missing
                ]
            )

        return ordered

    # ============================================================
    # PUBLIC READING ORDER API
    # ============================================================

    def assign_columns_and_reading_order(
        self,
        blocks: List[DocumentBlock],
        page_width: float = 612.0,
    ) -> List[DocumentBlock]:
        """
        Reconstruct logical reading order.

        Public API intentionally remains unchanged.
        """

        if not blocks:
            return []

        pages: Dict[
            int,
            List[DocumentBlock],
        ] = defaultdict(list)

        for block in blocks:
            pages[block.page].append(block)

        ordered_blocks: List[
            DocumentBlock
        ] = []

        global_order = 0

        for page_num in sorted(pages):

            page_blocks = pages[
                page_num
            ]

            ordered_page = (
                self._order_page(
                    page_blocks,
                    page_width,
                )
            )

            for block, column_id in ordered_page:

                block.column_id = column_id

                block.reading_order = (
                    global_order
                )

                global_order += 1

                ordered_blocks.append(
                    block
                )

        return ordered_blocks