"""Tests for Layout and Column Reading Order Reconstruction."""
import pytest
from src.document_models import DocumentBlock, BBox
from src.layout import LayoutEngine

def test_single_column_reading_order():
    engine = LayoutEngine()
    blocks = [
        DocumentBlock(id="b3", text="Third line", bbox=BBox(x0=50, y0=150, x1=300, y1=170)),
        DocumentBlock(id="b1", text="First line", bbox=BBox(x0=50, y0=50, x1=300, y1=70)),
        DocumentBlock(id="b2", text="Second line", bbox=BBox(x0=50, y0=100, x1=300, y1=120)),
    ]

    ordered = engine.assign_columns_and_reading_order(blocks)
    assert ordered[0].id == "b1"
    assert ordered[1].id == "b2"
    assert ordered[2].id == "b3"

def test_two_column_reading_order():
    engine = LayoutEngine()
    # Left column blocks (x centers ~ 150)
    col_left_1 = DocumentBlock(id="left_1", text="Left Col Top", bbox=BBox(x0=50, y0=100, x1=250, y1=120))
    col_left_2 = DocumentBlock(id="left_2", text="Left Col Bottom", bbox=BBox(x0=50, y0=200, x1=250, y1=220))
    
    # Right column blocks (x centers ~ 450)
    col_right_1 = DocumentBlock(id="right_1", text="Right Col Top", bbox=BBox(x0=350, y0=100, x1=550, y1=120))
    col_right_2 = DocumentBlock(id="right_2", text="Right Col Bottom", bbox=BBox(x0=350, y0=200, x1=550, y1=220))

    # Interleaved input
    interleaved = [col_right_1, col_left_1, col_right_2, col_left_2]

    ordered = engine.assign_columns_and_reading_order(interleaved, page_width=612.0)
    ordered_ids = [b.id for b in ordered]

    # Column 0 (left) should be read completely before Column 1 (right)
    assert ordered_ids == ["left_1", "left_2", "right_1", "right_2"]
