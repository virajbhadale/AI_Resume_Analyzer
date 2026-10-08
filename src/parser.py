"""
Production-grade unified document parser.

Responsibilities:
    PDF   -> digital text + layout metadata + OCR fallback
    DOCX  -> paragraphs + tables + headers/footers
    Image -> OCR
    TXT   -> deterministic text fixture support

Output:
    List[DocumentBlock]

The parser intentionally does NOT perform:
    - section classification
    - resume semantic extraction
    - ATS scoring
    - skill detection

Those responsibilities belong to downstream pipeline stages.
"""

import os
import re
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.document_models import DocumentBlock, BBox


# ============================================================
# EXCEPTIONS
# ============================================================

class ScannedDocumentError(Exception):
    """Raised when a document contains no usable digital text and OCR fails."""


class UnsupportedFormatError(Exception):
    """Raised when the supplied document format is unsupported."""


class DocumentParseError(Exception):
    """Raised when document parsing fails unexpectedly."""


# ============================================================
# PARSER
# ============================================================

class DocumentParser:
    """
    Unified ingestion layer for resume documents.

    Supported:
        .pdf
        .docx
        .txt
        .json
        .png
        .jpg
        .jpeg
        .webp
        .bmp
        .tiff

    Legacy .doc is intentionally NOT supported because python-docx
    cannot reliably parse the old binary Microsoft Word format.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".json",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
    }

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
    }

    TEXT_EXTENSIONS = {
        ".txt",
        ".json",
    }

    # Minimum useful digital text before OCR fallback is triggered.
    MIN_DIGITAL_TEXT_LENGTH = 40

    def __init__(
        self,
        min_digital_text_length: int = MIN_DIGITAL_TEXT_LENGTH,
    ):
        self.min_digital_text_length = max(
            1,
            int(min_digital_text_length)
        )

    # ========================================================
    # PUBLIC ENTRY POINT
    # ========================================================

    def parse(self, file_path: str) -> List[DocumentBlock]:
        """
        Parse a document and return normalized DocumentBlock objects.
        """

        if not file_path:
            raise ValueError("file_path cannot be empty.")

        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if not os.path.isfile(file_path):
            raise DocumentParseError(
                f"Path is not a regular file: {file_path}"
            )

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".doc":
            raise UnsupportedFormatError(
                "Legacy .doc files are not supported directly. "
                "Please convert the document to .docx or PDF."
            )

        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise UnsupportedFormatError(
                f"Unsupported file extension: {ext}. "
                f"Supported formats: {supported}"
            )

        try:
            if ext == ".pdf":
                blocks = self.parse_pdf(file_path)

            elif ext == ".docx":
                blocks = self.parse_docx(file_path)

            elif ext in self.IMAGE_EXTENSIONS:
                blocks = self.parse_image(file_path)

            elif ext in self.TEXT_EXTENSIONS:
                blocks = self.parse_text(file_path)

            else:
                raise UnsupportedFormatError(
                    f"Unsupported file extension: {ext}"
                )

        except (
            FileNotFoundError,
            UnsupportedFormatError,
            ScannedDocumentError,
        ):
            raise

        except Exception as exc:
            raise DocumentParseError(
                f"Failed to parse '{file_path}': {exc}"
            ) from exc

        if not blocks:
            raise ScannedDocumentError(
                f"No usable text blocks were extracted from: {file_path}"
            )

        return self._normalize_blocks(blocks)

    # ========================================================
    # PDF PARSING
    # ========================================================

    def parse_pdf(
        self,
        pdf_path: str
    ) -> List[DocumentBlock]:
        """
        Extract PDF text with real page dimensions and font metadata.

        If the PDF appears scanned or digital extraction is insufficient,
        OCR fallback is triggered.
        """

        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz
            except ImportError as exc:
                raise ImportError(
                    "PyMuPDF is required for PDF parsing. "
                    "Install with: pip install pymupdf"
                ) from exc

        blocks_out: List[DocumentBlock] = []
        total_text_len = 0

        try:
            doc = fitz.open(pdf_path)

            if doc.is_encrypted:
                if not doc.authenticate(""):
                    doc.close()
                    raise DocumentParseError(
                        "PDF is password protected and cannot be parsed."
                    )

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1

                rect = page.rect
                page_width = float(rect.width)
                page_height = float(rect.height)

                page_dict = page.get_text("dict")

                raw_blocks = page_dict.get("blocks", [])

                for block_idx, raw_block in enumerate(raw_blocks):

                    # Text blocks contain "lines".  We intentionally emit
                    # one DocumentBlock per visual line rather than merging
                    # an entire PDF text block.  Resume layouts commonly put
                    # role/company/date/bullet lines in the same PDF block;
                    # preserving line geometry makes downstream section,
                    # experience and column reconstruction much more reliable.
                    lines = raw_block.get("lines")
                    if not lines:
                        continue

                    for line_idx, line in enumerate(lines):
                        line_parts: List[str] = []
                        max_font_size = 0.0
                        is_bold = False
                        is_italic = False
                        font_names: List[str] = []

                        for span in line.get("spans", []):
                            span_text = str(span.get("text", "") or "")
                            if not span_text:
                                continue

                            line_parts.append(span_text)

                            font_size = float(
                                span.get("size", 10.0) or 10.0
                            )
                            max_font_size = max(max_font_size, font_size)

                            font_name = str(
                                span.get("font", "") or ""
                            ).strip()
                            if font_name:
                                font_names.append(font_name)

                            flags = int(span.get("flags", 0) or 0)
                            if flags & 16 or "bold" in font_name.lower():
                                is_bold = True
                            if flags & 2 or "italic" in font_name.lower():
                                is_italic = True

                        line_text = "".join(line_parts).strip()
                        if not line_text:
                            continue

                        total_text_len += len(line_text)

                        # Prefer the line's real geometry.  Fall back to the
                        # parent block geometry for malformed PDF line data.
                        line_bbox_raw = line.get("bbox") or raw_block.get(
                            "bbox", [0.0, 0.0, 0.0, 0.0]
                        )
                        if len(line_bbox_raw) != 4:
                            line_bbox_raw = [0.0, 0.0, 0.0, 0.0]

                        bbox = BBox(
                            x0=float(line_bbox_raw[0]),
                            y0=float(line_bbox_raw[1]),
                            x1=float(line_bbox_raw[2]),
                            y1=float(line_bbox_raw[3]),
                        )

                        font_name = (
                            Counter(font_names).most_common(1)[0][0]
                            if font_names
                            else "default"
                        )

                        word_count = len(line_text.split())
                        is_heading = (
                            1 <= word_count <= 8
                            and (
                                max_font_size >= 12.0
                                or is_bold
                                or (
                                    line_text.isupper()
                                    and any(c.isalpha() for c in line_text)
                                )
                            )
                        )

                        blocks_out.append(
                            DocumentBlock(
                                id=(
                                    f"p{page_num}_b{block_idx}_l{line_idx}_"
                                    f"{uuid.uuid4().hex[:8]}"
                                ),
                                page=page_num,
                                page_width=page_width,
                                page_height=page_height,
                                text=line_text,
                                bbox=bbox,
                                font_size=(
                                    max_font_size
                                    if max_font_size > 0
                                    else 10.0
                                ),
                                font_name=font_name,
                                is_bold=is_bold,
                                is_italic=is_italic,
                                is_heading=is_heading,
                                source_type="pdf",
                                extraction_method="pdf_text",
                                block_type="text",
                                confidence=0.98,
                            )
                        )

            doc.close()

        except DocumentParseError:
            raise

        except Exception as exc:
            # Direct extraction failed.
            print(
                "[WARN] PDF digital extraction failed: "
                f"{exc}"
            )

            blocks_out = []
            total_text_len = 0

        # ----------------------------------------------------
        # OCR FALLBACK
        # ----------------------------------------------------

        if total_text_len < self.min_digital_text_length:

            print(
                "[INFO] Insufficient digital PDF text "
                f"({total_text_len} characters). "
                "Activating OCR fallback..."
            )

            try:
                from src.ocr import (
                    ocr_pdf,
                    text_to_document_blocks,
                )

                page_texts = ocr_pdf(pdf_path)

                ocr_blocks: List[DocumentBlock] = []

                for page_num, text_content in page_texts:

                    if not text_content.strip():
                        continue

                    page_blocks = text_to_document_blocks(
                        text_content,
                        page_num=page_num,
                        prefix=f"scan_p{page_num}",
                    )

                    ocr_blocks.extend(page_blocks)

                if ocr_blocks:
                    return ocr_blocks

            except Exception as exc:
                raise ScannedDocumentError(
                    "PDF appears to be scanned or contains "
                    "insufficient digital text, and OCR failed: "
                    f"{exc}"
                ) from exc

            raise ScannedDocumentError(
                "PDF contains insufficient extractable text "
                "and OCR returned no usable content."
            )

        # Merge PDF fragments that belong to the same visual line. Some
        # designer-generated resumes store each word as a separate PDF
        # text block. Keeping those fragments separate makes section and
        # sentence extraction unnecessarily brittle.
        blocks_out = self._merge_pdf_line_fragments(blocks_out)

        # Add document-level page metadata.
        self._attach_pdf_metadata(
            blocks_out,
            pdf_path
        )

        return blocks_out

    # ========================================================
    # PDF LINE FRAGMENT RECONSTRUCTION
    # ========================================================

    def _merge_pdf_line_fragments(
        self,
        blocks: List[DocumentBlock],
    ) -> List[DocumentBlock]:
        """
        Merge adjacent PDF text fragments that visually belong to one line.

        Some resume generators emit every word as a separate PDF text block.
        We merge only when the evidence is strong:
            - same page
            - nearly identical vertical position
            - compatible typography
            - small horizontal gap
            - neither block looks like a bullet boundary

        This preserves columns while reconstructing natural text such as:
            "analysis, machine learning, and"
        from four separate PDF fragments.
        """

        if len(blocks) < 2:
            return blocks

        by_page: Dict[int, List[DocumentBlock]] = {}
        for block in blocks:
            by_page.setdefault(block.page, []).append(block)

        merged_all: List[DocumentBlock] = []

        for page_num in sorted(by_page):
            page_blocks = sorted(
                by_page[page_num],
                key=lambda b: (b.bbox.y0, b.bbox.x0),
            )

            current: Optional[DocumentBlock] = None

            for block in page_blocks:
                if current is None:
                    current = block
                    continue

                y_delta = abs(current.bbox.center_y - block.bbox.center_y)
                x_gap = block.bbox.x0 - current.bbox.x1
                font_delta = abs(current.font_size - block.font_size)

                bullet_boundary = (
                    self._is_bullet_text(current.text)
                    or self._is_bullet_text(block.text)
                )

                compatible = (
                    y_delta <= 2.5
                    and 0.0 <= x_gap <= 24.0
                    and font_delta <= 1.5
                    and current.is_bold == block.is_bold
                    and current.is_italic == block.is_italic
                    and not bullet_boundary
                    and len(current.text) + len(block.text) <= 240
                )

                if compatible:
                    current.text = (
                        current.text.rstrip()
                        + " "
                        + block.text.lstrip()
                    ).strip()
                    current.bbox = BBox(
                        x0=min(current.bbox.x0, block.bbox.x0),
                        y0=min(current.bbox.y0, block.bbox.y0),
                        x1=max(current.bbox.x1, block.bbox.x1),
                        y1=max(current.bbox.y1, block.bbox.y1),
                    )
                    current.confidence = min(
                        current.confidence,
                        block.confidence,
                    )
                else:
                    merged_all.append(current)
                    current = block

            if current is not None:
                merged_all.append(current)

        return merged_all

    # ========================================================
    # PDF METADATA
    # ========================================================

    def _attach_pdf_metadata(
        self,
        blocks: List[DocumentBlock],
        pdf_path: str
    ) -> None:
        """
        Attach conservative header/footer semantic hints.

        We do NOT delete these blocks here because some resumes
        intentionally place important information in headers/footers.
        Downstream layout/section logic can decide what to do.
        """

        if not blocks:
            return

        pages: Dict[int, List[DocumentBlock]] = {}

        for block in blocks:
            pages.setdefault(block.page, []).append(block)

        # Detect repeated exact text across pages.
        text_pages: Dict[str, set] = {}

        for block in blocks:
            normalized = self._normalize_text(block.text)

            if not normalized:
                continue

            text_pages.setdefault(normalized, set()).add(
                block.page
            )

        for block in blocks:

            normalized = self._normalize_text(block.text)

            # Repeated content across multiple pages is often
            # a header/footer, but we only mark it semantically.
            repeated = (
                len(text_pages.get(normalized, set())) >= 2
            )

            if repeated:

                page_height = block.page_height

                if page_height > 0:

                    relative_y0 = (
                        block.bbox.y0 / page_height
                    )

                    relative_y1 = (
                        block.bbox.y1 / page_height
                    )

                    if relative_y0 <= 0.08:
                        block.semantic_label = "HEADER_CANDIDATE"
                        block.is_header_candidate = True

                    elif relative_y1 >= 0.92:
                        block.semantic_label = "FOOTER_CANDIDATE"
                        block.is_footer_candidate = True

    # ========================================================
    # DOCX PARSING
    # ========================================================

    def parse_docx(
        self,
        docx_path: str
    ) -> List[DocumentBlock]:
        """
        Parse DOCX paragraphs, tables, headers and footers.

        DOCX does not expose PDF-like coordinates naturally,
        so virtual coordinates are used to preserve ordering.
        """

        try:
            import docx
        except ImportError as exc:
            raise ImportError(
                "python-docx is required for DOCX parsing. "
                "Install with: pip install python-docx"
            ) from exc

        doc = docx.Document(docx_path)

        blocks_out: List[DocumentBlock] = []

        virtual_y = 50.0
        sequence = 0

        # ----------------------------------------------------
        # HEADERS
        # ----------------------------------------------------

        for section_idx, section in enumerate(
            doc.sections
        ):
            for p_idx, paragraph in enumerate(
                section.header.paragraphs
            ):

                block = self._docx_paragraph_to_block(
                    paragraph=paragraph,
                    page=1,
                    virtual_y=virtual_y,
                    block_id=(
                        f"header_s{section_idx}_"
                        f"{p_idx}_{uuid.uuid4().hex[:8]}"
                    ),
                )

                if block:
                    block.semantic_label = "HEADER_CANDIDATE"
                    blocks_out.append(block)

                    virtual_y += block.bbox.height + 8.0

        # ----------------------------------------------------
        # BODY PARAGRAPHS + TABLES
        # ----------------------------------------------------

        body_items = self._iter_docx_body_items(doc)

        for item_type, item in body_items:

            if item_type == "paragraph":

                block = self._docx_paragraph_to_block(
                    paragraph=item,
                    page=1,
                    virtual_y=virtual_y,
                    block_id=(
                        f"docx_p{sequence}_"
                        f"{uuid.uuid4().hex[:8]}"
                    ),
                )

                sequence += 1

                if block:
                    blocks_out.append(block)
                    virtual_y += block.bbox.height + 8.0

            elif item_type == "table":

                table_blocks = self._docx_table_to_blocks(
                    table=item,
                    page=1,
                    start_y=virtual_y,
                    table_index=sequence,
                )

                sequence += 1

                blocks_out.extend(table_blocks)

                if table_blocks:
                    virtual_y = (
                        max(
                            b.bbox.y1
                            for b in table_blocks
                        )
                        + 8.0
                    )

        # ----------------------------------------------------
        # FOOTERS
        # ----------------------------------------------------

        for section_idx, section in enumerate(
            doc.sections
        ):
            for p_idx, paragraph in enumerate(
                section.footer.paragraphs
            ):

                block = self._docx_paragraph_to_block(
                    paragraph=paragraph,
                    page=1,
                    virtual_y=virtual_y,
                    block_id=(
                        f"footer_s{section_idx}_"
                        f"{p_idx}_{uuid.uuid4().hex[:8]}"
                    ),
                )

                if block:
                    block.semantic_label = "FOOTER_CANDIDATE"
                    blocks_out.append(block)

                    virtual_y += block.bbox.height + 8.0

        return blocks_out

    # ========================================================
    # DOCX BODY ORDER
    # ========================================================

    def _iter_docx_body_items(
        self,
        doc
    ):
        """
        Preserve the actual XML order of paragraphs and tables.

        This is important because:
            paragraph -> table -> paragraph

        should NOT become:

            all paragraphs -> all tables
        """

        from docx.document import Document as _Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        from docx.oxml.text.paragraph import CT_P
        from docx.oxml.table import CT_Tbl

        parent = doc.element.body

        for child in parent.iterchildren():

            if isinstance(child, CT_P):
                yield "paragraph", Paragraph(
                    child,
                    parent
                )

            elif isinstance(child, CT_Tbl):
                yield "table", Table(
                    child,
                    parent
                )

    # ========================================================
    # DOCX PARAGRAPH
    # ========================================================

    def _docx_paragraph_to_block(
        self,
        paragraph,
        page: int,
        virtual_y: float,
        block_id: str,
    ) -> Optional[DocumentBlock]:

        text = paragraph.text.strip()

        if not text:
            return None

        style_name = ""

        try:
            if paragraph.style:
                style_name = (
                    paragraph.style.name.lower()
                )
        except Exception:
            style_name = ""

        is_bold = any(
            run.bold is True
            for run in paragraph.runs
        )

        is_italic = any(
            run.italic is True
            for run in paragraph.runs
        )

        # Determine actual run font size when available.
        font_sizes = []

        for run in paragraph.runs:
            if run.font.size:
                try:
                    font_sizes.append(
                        run.font.size.pt
                    )
                except Exception:
                    pass

        if font_sizes:
            font_size = max(font_sizes)
        elif (
            "heading" in style_name
            or "title" in style_name
        ):
            font_size = 14.0
        else:
            font_size = 10.5

        word_count = len(text.split())

        is_heading = (
            "heading" in style_name
            or "title" in style_name
            or (
                is_bold
                and word_count <= 8
            )
        )

        height = (
            18.0
            if is_heading
            else max(
                14.0,
                14.0 * (
                    1
                    + len(text) // 90
                )
            )
        )

        bbox = BBox(
            x0=54.0,
            y0=virtual_y,
            x1=558.0,
            y1=virtual_y + height,
        )

        return DocumentBlock(
            id=block_id,
            page=page,
            text=text,
            bbox=bbox,

            page_width=612.0,
            page_height=792.0,

            source_type="docx",
            extraction_method="docx_text",
            block_type="text",

            font_size=float(font_size),
            font_name="docx",
            is_bold=is_bold,
            is_italic=is_italic,
            is_heading=is_heading,
            confidence=0.96,
        )

    # ========================================================
    # DOCX TABLES
    # ========================================================

    def _docx_table_to_blocks(
        self,
        table,
        page: int,
        start_y: float,
        table_index: int,
    ) -> List[DocumentBlock]:

        blocks: List[DocumentBlock] = []

        virtual_y = start_y

        for row_idx, row in enumerate(
            table.rows
        ):

            for cell_idx, cell in enumerate(
                row.cells
            ):

                text = cell.text.strip()

                if not text:
                    continue

                # Preserve cell content while avoiding
                # duplicate paragraphs inside a cell.
                lines = [
                    line.strip()
                    for line in text.splitlines()
                    if line.strip()
                ]

                if not lines:
                    continue

                cell_text = "\n".join(lines)

                height = max(
                    16.0,
                    16.0 * len(lines)
                )

                bbox = BBox(
                    x0=54.0,
                    y0=virtual_y,
                    x1=558.0,
                    y1=virtual_y + height,
                )

                block = DocumentBlock(
                    id=(
                        f"docx_t{table_index}_"
                        f"r{row_idx}_"
                        f"c{cell_idx}_"
                        f"{uuid.uuid4().hex[:8]}"
                    ),
                    page=page,
                    text=cell_text,
                    bbox=bbox,

                    page_width=612.0,
                    page_height=792.0,

                    source_type="docx",
                    extraction_method="docx_table",
                    block_type="table_cell",

                    font_size=10.5,
                    font_name="docx-table",
                    confidence=0.94,
                )

                blocks.append(block)

                virtual_y += height + 4.0

        return blocks

    # ========================================================
    # IMAGE PARSING
    # ========================================================

    def parse_image(
        self,
        image_path: str
    ) -> List[DocumentBlock]:
        """
        Process image resumes through the existing OCR engine.
        """

        from src.ocr import (
            ocr_image_path,
            text_to_document_blocks,
        )

        print(
            f"[INFO] Processing image resume via OCR: "
            f"{image_path}"
        )

        text = ocr_image_path(image_path)

        if not text.strip():
            raise ScannedDocumentError(
                "OCR extracted no usable text from image."
            )

        return text_to_document_blocks(
            text,
            page_num=1,
            prefix="img",
        )

    # ========================================================
    # TEXT / JSON PARSING
    # ========================================================

    def parse_text(
        self,
        text_path: str
    ) -> List[DocumentBlock]:
        """
        Parse TXT/JSON fixtures into deterministic blocks.

        JSON is treated as textual fixture input at this layer.
        Semantic JSON parsing belongs elsewhere.
        """

        with open(
            text_path,
            "r",
            encoding="utf-8"
        ) as f:
            content = f.read()

        if not content.strip():
            raise ScannedDocumentError(
                f"Text document is empty: {text_path}"
            )

        lines = content.splitlines()

        blocks_out: List[DocumentBlock] = []

        virtual_y = 50.0

        for idx, line in enumerate(lines):

            stripped = line.strip()

            if not stripped:
                continue

            words = stripped.split()

            is_upper = (
                stripped.isupper()
                and len(stripped) >= 3
            )

            is_heading = (
                len(words) <= 8
                and (
                    is_upper
                    or stripped.endswith(":")
                )
            )

            font_size = (
                13.0
                if is_heading
                else 10.0
            )

            height = (
                18.0
                if is_heading
                else max(
                    16.0,
                    16.0 * (
                        1
                        + len(stripped) // 100
                    )
                )
            )

            bbox = BBox(
                x0=54.0,
                y0=virtual_y,
                x1=558.0,
                y1=virtual_y + height,
            )

            block = DocumentBlock(
                id=(
                    f"txt_{idx}_"
                    f"{uuid.uuid4().hex[:8]}"
                ),
                page=1,
                text=stripped,
                bbox=bbox,
                font_size=font_size,
                font_name="text",
                is_bold=is_heading,
                is_heading=is_heading,
                confidence=0.99,
            )

            blocks_out.append(block)

            virtual_y += height + 6.0

        return blocks_out

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def _normalize_blocks(
        self,
        blocks: List[DocumentBlock]
    ) -> List[DocumentBlock]:
        """
        Normalize text while preserving semantic information.
        """

        normalized_blocks: List[DocumentBlock] = []

        for block in blocks:

            text = self._normalize_text(
                block.text,
                preserve_newlines=True
            )

            if not text:
                continue

            block.text = text

            # Clamp invalid confidence values.
            block.confidence = max(
                0.0,
                min(
                    1.0,
                    float(block.confidence)
                )
            )

            normalized_blocks.append(block)

        return normalized_blocks

    @staticmethod
    def _is_bullet_text(text: str) -> bool:
        """
        Return True when text appears to start with a bullet/list marker.
        Handles normal Unicode bullets and common mojibake variants.
        """

        if not text:
            return False

        stripped = text.lstrip()

        bullet_prefixes = (
            "•",
            "◦",
            "▪",
            "▫",
            "‣",
            "⁃",
            "∙",
            "●",
            "○",
            "◆",
            "◇",
            "* ",
            "- ",
            "– ",
            "— ",
            "â€¢",
            "â€“ ",
            "â€” ",
        )

        return stripped.startswith(bullet_prefixes)

    @staticmethod
    def _normalize_text(
        text: str,
        preserve_newlines: bool = False
    ) -> str:

        if not text:
            return ""

        # Normalize non-breaking spaces
        text = text.replace("\u00a0", " ")

        # Normalize Unicode punctuation
        text = text.replace("\u2011", "-")  # non-breaking hyphen
        text = text.replace("\u2013", "-")  # en dash
        text = text.replace("\u2014", "-")  # em dash

        # Normalize common bullet characters
        text = text.replace("\u2022", "-")
        text = text.replace("\u25CF", "-")
        text = text.replace("\u25AA", "-")
        text = text.replace("\u25AB", "-")
        text = text.replace("\u2219", "-")
        text = text.replace("\uf0b7", "-")

        # Normalize common mojibake representations
        text = text.replace("â€¢", "-")
        text = text.replace("â€“", "-")
        text = text.replace("â€”", "-")

        if preserve_newlines:
            lines = [
                re.sub(
                    r"[ \t]+",
                    " ",
                    line
                ).strip()
                for line in text.splitlines()
            ]

            return "\n".join(
                line
                for line in lines
                if line
            ).strip()

        return re.sub(
            r"\s+",
            " ",
            text
        ).strip()