"""
Tests for Document Parser.

Run:
    pytest -q tests/test_parser.py
"""

import json
import os

import pytest

from src.parser import (
    DocumentParser,
    UnsupportedFormatError,
    ScannedDocumentError,
    DocumentParseError,
)


# ============================================================
# HELPERS
# ============================================================

def create_file(tmp_path, filename, content="sample content"):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


# ============================================================
# 1. BASIC TEXT PARSING
# ============================================================

def test_text_fixture_parsing():
    parser = DocumentParser()

    sample_file = "data/samples/sample_resume_1.txt"

    blocks = parser.parse(sample_file)

    assert len(blocks) > 10

    # First block should contain candidate name.
    assert "ALEXANDER CHEN" in blocks[0].text


# ============================================================
# 2. TEXT PARSING CREATES DOCUMENT BLOCKS
# ============================================================

def test_text_parsing_creates_document_blocks(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "resume.txt",
        """JOHN DOE

Education
B.Tech Computer Science
ABC University
2020 - 2024

Skills
Python
SQL
""",
    )

    blocks = parser.parse(path)

    assert blocks
    assert all(block.text for block in blocks)
    assert all(block.page >= 1 for block in blocks)


# ============================================================
# 3. TEXT HEADINGS ARE DETECTED
# ============================================================

def test_text_heading_detection(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "resume.txt",
        """JOHN DOE
EDUCATION
B.Tech Computer Science
ABC University
EXPERIENCE
Software Engineer
""",
    )

    blocks = parser.parse(path)

    headings = [
        block.text
        for block in blocks
        if block.is_heading
    ]

    assert "EDUCATION" in headings
    assert "EXPERIENCE" in headings


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def test_text_normalization(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "resume.txt",
        "John\u00a0Doe\nPython\u2013Developer\nMachine\u2014Learning",
    )

    blocks = parser.parse(path)

    text = "\n".join(block.text for block in blocks)

    assert "\u00a0" not in text
    assert "\u2013" not in text
    assert "\u2014" not in text

    assert "Python-Developer" in text
    assert "Machine-Learning" in text


# ============================================================
# 5. EMPTY TEXT FILE
# ============================================================

def test_empty_text_file_raises_scanned_document_error(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "empty.txt",
        "",
    )

    with pytest.raises(ScannedDocumentError):
        parser.parse(path)


# ============================================================
# 6. UNSUPPORTED FORMAT
# ============================================================

def test_unsupported_format():
    parser = DocumentParser()

    test_dummy = "test_file.xyz"

    with open(test_dummy, "w", encoding="utf-8") as f:
        f.write("dummy")

    try:
        with pytest.raises(UnsupportedFormatError):
            parser.parse(test_dummy)
    finally:
        if os.path.exists(test_dummy):
            os.remove(test_dummy)


# ============================================================
# 7. LEGACY DOC FORMAT
# ============================================================

def test_legacy_doc_is_rejected(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "resume.doc",
        "dummy",
    )

    with pytest.raises(UnsupportedFormatError):
        parser.parse(path)


# ============================================================
# 8. NONEXISTENT FILE
# ============================================================

def test_nonexistent_file_raises(tmp_path):
    parser = DocumentParser()

    path = str(tmp_path / "does_not_exist.txt")

    with pytest.raises(FileNotFoundError):
        parser.parse(path)


# ============================================================
# 9. JSON TEXT FIXTURE
# ============================================================

def test_json_parsing(tmp_path):
    parser = DocumentParser()

    data = {
        "name": "John Doe",
        "skills": ["Python", "SQL"],
    }

    path = tmp_path / "resume.json"

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    blocks = parser.parse(str(path))

    assert blocks
    combined = " ".join(block.text for block in blocks)

    assert "John Doe" in combined
    assert "Python" in combined
    assert "SQL" in combined


# ============================================================
# 10. CONFIDENCE VALUES ARE VALID
# ============================================================

def test_block_confidence_is_normalized(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "resume.txt",
        """JOHN DOE
Python Developer
Skills
Python
SQL
""",
    )

    blocks = parser.parse(path)

    assert blocks

    for block in blocks:
        assert 0.0 <= block.confidence <= 1.0


# ============================================================
# 11. BLOCK TEXT IS NOT EMPTY
# ============================================================

def test_parser_does_not_return_empty_blocks(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "resume.txt",
        """John Doe


Python


Developer
""",
    )

    blocks = parser.parse(path)

    assert blocks

    for block in blocks:
        assert block.text.strip()


# ============================================================
# 12. MULTI-LINE TEXT PRESERVES ORDER
# ============================================================

def test_text_block_order_is_preserved(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "resume.txt",
        """John Doe
Education
B.Tech
ABC University
Experience
Software Engineer
""",
    )

    blocks = parser.parse(path)

    texts = [block.text for block in blocks]

    assert texts.index("John Doe") < texts.index("Education")
    assert texts.index("Education") < texts.index("B.Tech")
    assert texts.index("B.Tech") < texts.index("ABC University")
    assert texts.index("ABC University") < texts.index("Experience")


# ============================================================
# 13. PDF LINE FRAGMENT RECONSTRUCTION
# ============================================================

def test_pdf_line_fragment_reconstruction_on_real_resume_fixture():
    """
    Designer PDFs that split words into separate blocks should be rejoined.
    """

    pdf_path = "/mnt/data/project_work/test_inputs/VIRAJ (3).pdf"

    if not os.path.exists(pdf_path):
        pytest.skip(
            "Real PDF fixture not available in this environment"
        )

    parser = DocumentParser()

    blocks = parser.parse(pdf_path)

    texts = [b.text for b in blocks]

    assert "analysis, machine learning, and" in texts


# ============================================================
# 14. PDF PARSING BASIC VALIDATION
# ============================================================

def test_pdf_parser_returns_blocks_when_fixture_exists():
    pdf_path = "/mnt/data/project_work/test_inputs/VIRAJ (3).pdf"

    if not os.path.exists(pdf_path):
        pytest.skip(
            "Real PDF fixture not available in this environment"
        )

    parser = DocumentParser()

    blocks = parser.parse(pdf_path)

    assert blocks

    for block in blocks:
        assert block.text.strip()
        assert block.page >= 1
        assert 0.0 <= block.confidence <= 1.0


# ============================================================
# 15. PDF BLOCK GEOMETRY
# ============================================================

def test_pdf_blocks_have_valid_geometry():
    pdf_path = "/mnt/data/project_work/test_inputs/VIRAJ (3).pdf"

    if not os.path.exists(pdf_path):
        pytest.skip(
            "Real PDF fixture not available in this environment"
        )

    parser = DocumentParser()

    blocks = parser.parse(pdf_path)

    assert blocks

    for block in blocks:
        assert block.bbox.x1 >= block.bbox.x0
        assert block.bbox.y1 >= block.bbox.y0


# ============================================================
# 16. DOCX PARSING
# ============================================================

def test_docx_parsing(tmp_path):
    docx = pytest.importorskip("docx")

    document = docx.Document()

    document.add_paragraph("JOHN DOE")
    document.add_paragraph("Education")
    document.add_paragraph("B.Tech Computer Science")
    document.add_paragraph("ABC University")

    path = tmp_path / "resume.docx"

    document.save(path)

    parser = DocumentParser()

    blocks = parser.parse(str(path))

    assert blocks

    texts = [block.text for block in blocks]

    assert "JOHN DOE" in texts
    assert "Education" in texts
    assert "B.Tech Computer Science" in texts
    assert "ABC University" in texts


# ============================================================
# 17. DOCX TABLE PARSING
# ============================================================

def test_docx_table_parsing(tmp_path):
    docx = pytest.importorskip("docx")

    document = docx.Document()

    table = document.add_table(
        rows=2,
        cols=2,
    )

    table.cell(0, 0).text = "Degree"
    table.cell(0, 1).text = "Institution"

    table.cell(1, 0).text = "B.Tech"
    table.cell(1, 1).text = "ABC University"

    path = tmp_path / "table_resume.docx"

    document.save(path)

    parser = DocumentParser()

    blocks = parser.parse(str(path))

    assert blocks

    combined = "\n".join(
        block.text
        for block in blocks
    )

    assert "Degree" in combined
    assert "Institution" in combined
    assert "B.Tech" in combined
    assert "ABC University" in combined


# ============================================================
# 18. IMAGE PARSING INTERFACE
# ============================================================

def test_image_parser_method_exists():
    parser = DocumentParser()

    assert callable(parser.parse_image)


# ============================================================
# 19. SUPPORTED EXTENSIONS
# ============================================================

def test_supported_extensions_are_defined():
    parser = DocumentParser()

    assert parser.SUPPORTED_EXTENSIONS

    for extension in parser.SUPPORTED_EXTENSIONS:
        assert extension.startswith(".")


# ============================================================
# 20. NORMALIZATION_REMOVES_EMPTY_BLOCKS
# ============================================================

def test_normalization_removes_empty_blocks():
    parser = DocumentParser()

    blocks = []

    normalized = parser._normalize_blocks(blocks)

    assert normalized == []


# ============================================================
# 21. PDF PAGE METADATA
# ============================================================

def test_pdf_blocks_have_page_metadata():
    pdf_path = "/mnt/data/project_work/test_inputs/VIRAJ (3).pdf"

    if not os.path.exists(pdf_path):
        pytest.skip(
            "Real PDF fixture not available in this environment"
        )

    parser = DocumentParser()

    blocks = parser.parse(pdf_path)

    assert blocks

    for block in blocks:
        assert block.page_width > 0
        assert block.page_height > 0
        assert block.source_type == "pdf"
        assert block.extraction_method == "pdf_text"


# ============================================================
# 22. RELATIVE PAGE GEOMETRY
# ============================================================

def test_pdf_blocks_have_valid_relative_geometry():
    pdf_path = "/mnt/data/project_work/test_inputs/VIRAJ (3).pdf"

    if not os.path.exists(pdf_path):
        pytest.skip(
            "Real PDF fixture not available in this environment"
        )

    parser = DocumentParser()

    blocks = parser.parse(pdf_path)

    assert blocks

    for block in blocks:
        assert 0.0 <= block.relative_x0 <= 1.0
        assert 0.0 <= block.relative_x1 <= 1.0
        assert 0.0 <= block.relative_y0 <= 1.0
        assert 0.0 <= block.relative_y1 <= 1.0
        
# ============================================================
# 23. UNICODE BULLET NORMALIZATION
# ============================================================

def test_unicode_bullet_normalization(tmp_path):
    parser = DocumentParser()

    path = create_file(
        tmp_path,
        "bullet_resume.txt",
        "â€¢ Python\n"
        "â€“ Machine Learning\n"
        "Data Engineering\n",
    )

    blocks = parser.parse(path)

    combined = "\n".join(
        block.text
        for block in blocks
    )

    assert "â€¢" not in combined
    assert "â€“" not in combined
    
                        