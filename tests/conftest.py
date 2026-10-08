import uuid

import pytest

from src.document_models import DocumentBlock


@pytest.fixture
def make_document_block():
    def _make_document_block(
        text: str,
        section: str | None = None,
        *,
        is_bold: bool = False,
        is_heading: bool = False,
        page: int = 1,
        column_id: int = 0,
        reading_order: int = 0,
    ) -> DocumentBlock:

        return DocumentBlock(
            id=f"test_block_{uuid.uuid4().hex[:8]}",
            text=text,
            section=section,
            page=page,
            is_bold=is_bold,
            is_heading=is_heading,
            column_id=column_id,
            reading_order=reading_order,
            source_type="text",
            extraction_method="test",
        )

    return _make_document_block