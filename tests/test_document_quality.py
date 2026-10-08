from src.document_models import (
    BBox,
    DocumentBlock,
)

from src.quality.analyzer import (
    DocumentQualityAnalyzer,
)


def make_block(
    text: str,
    page: int = 1,
) -> DocumentBlock:

    return DocumentBlock(
        id=f"block-{page}",
        page=page,
        text=text,
        bbox=BBox(
            x0=0,
            y0=0,
            x1=500,
            y1=100,
        ),
        page_width=500,
        page_height=700,
        extraction_method="pdf_text",
        source_type="pdf",
    )


def test_quality_analyzer_basic_document():

    blocks = [
        make_block(
            "Senior Data Scientist\n"
            "Built machine learning models "
            "for production applications."
        )
    ]

    report = DocumentQualityAnalyzer().analyze(
        blocks
    )

    assert report.page_count == 1
    assert report.block_count == 1
    assert report.total_text_length > 0
    assert report.text_coverage == 1.0
    assert report.requires_ocr is False
    assert report.overall_confidence > 0


def test_quality_analyzer_empty_document():

    report = DocumentQualityAnalyzer().analyze(
        []
    )

    assert report.overall_confidence == 0.0
    assert len(report.errors) == 1


def test_quality_analyzer_detects_mojibake():

    blocks = [
        make_block(
            "Senior Engineer â€“ Data Platform"
        )
    ]

    report = DocumentQualityAnalyzer().analyze(
        blocks
    )

    assert report.has_encoding_issues is True


def test_quality_analyzer_detects_low_text_page():

    blocks = [
        make_block(
            "Resume",
            page=1,
        )
    ]

    report = DocumentQualityAnalyzer().analyze(
        blocks
    )

    assert report.pages[0].likely_empty is True