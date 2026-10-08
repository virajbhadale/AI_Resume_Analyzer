from src.document_models import DocumentBlock, BBox
from src.extraction.contact import ContactExtractor


def test_name_can_be_reconstructed_from_adjacent_header_lines():
    blocks = [
        DocumentBlock(
            id="name1",
            text="Viraj",
            bbox=BBox(x0=80, y0=40, x1=150, y1=65),
            font_size=22,
            reading_order=0,
        ),
        DocumentBlock(
            id="name2",
            text="Bhadale",
            bbox=BBox(x0=80, y0=68, x1=160, y1=93),
            font_size=22,
            reading_order=1,
        ),
        DocumentBlock(
            id="title",
            text="Data Scientist",
            bbox=BBox(x0=80, y0=98, x1=180, y1=112),
            font_size=12,
            reading_order=2,
        ),
    ]

    contact = ContactExtractor().extract(blocks)
    assert contact.name == "Viraj Bhadale"
