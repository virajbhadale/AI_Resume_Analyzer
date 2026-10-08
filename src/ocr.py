"""AI OCR Engine for scanned PDFs and image resumes."""
import os
import subprocess
import tempfile
from typing import List, Tuple
from src.document_models import DocumentBlock, BBox

def ocr_image_path(image_path: str) -> str:
    """Executes OCR on an image file using the server-side Gemini 3.8 Flash model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found for OCR: {image_path}")

    # Use node src/ocr_engine.js
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ocr_engine.js"))
    cmd = ["node", script_path, os.path.abspath(image_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"OCR execution failed: {result.stderr or result.stdout}")

    return result.stdout.strip()

def ocr_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """Renders each page of a scanned PDF as an image and performs OCR on each page."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open(pdf_path)
    page_texts: List[Tuple[int, str]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            pix = page.get_pixmap(dpi=150)
            page_img_path = os.path.join(tmp_dir, f"page_{page_num}.png")
            pix.save(page_img_path)

            extracted_text = ocr_image_path(page_img_path)
            page_texts.append((page_num, extracted_text))

    doc.close()
    return page_texts

def text_to_document_blocks(text: str, page_num: int = 1, prefix: str = "ocr") -> List[DocumentBlock]:
    """Converts OCR line-by-line text into DocumentBlock structures with estimated geometries."""
    blocks: List[DocumentBlock] = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    virtual_y = 50.0

    for idx, line in enumerate(lines):
        # Heading heuristic: Short uppercase or capital line with <= 5 words
        words = line.split()
        is_short = len(words) <= 5
        is_upper = line.isupper() and len(line) >= 3
        is_heading = is_short and (is_upper or line.endswith(":"))

        font_size = 14.0 if is_heading else (18.0 if idx == 0 and len(words) <= 4 else 10.5)
        is_bold = is_heading or (idx == 0 and len(words) <= 4)
        height = 18.0 if is_heading else 14.0

        bbox = BBox(x0=54.0, y0=virtual_y, x1=558.0, y1=virtual_y + height)
        virtual_y += height + 6.0

        block = DocumentBlock(
            id=f"{prefix}_p{page_num}_{idx}",
            page=page_num,
            text=line,
            bbox=bbox,
            font_size=font_size,
            is_bold=is_bold,
            is_heading=is_heading,
            confidence=0.92
        )
        blocks.append(block)

    return blocks
