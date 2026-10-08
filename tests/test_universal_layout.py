"""
Universal geometry/layout behavior tests.

These tests describe layout behavior, not a particular resume template.
"""

from src.document_models import BBox, DocumentBlock
from src.layout import LayoutEngine


def block(
    block_id,
    x0,
    y0,
    x1,
    y1,
    text=None,
    page=1,
):
    return DocumentBlock(
        id=block_id,
        text=text or block_id,
        page=page,
        page_width=600,
        page_height=800,
        bbox=BBox(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        ),
        source_type="test",
        extraction_method="fixture",
    )


def test_three_column_layout_is_read_left_to_right_by_region():

    blocks = [
        block(
            "c2_bottom",
            220,
            250,
            380,
            270,
        ),
        block(
            "c1_top",
            30,
            100,
            180,
            120,
        ),
        block(
            "c3_top",
            420,
            100,
            570,
            120,
        ),
        block(
            "c2_top",
            220,
            100,
            380,
            120,
        ),
        block(
            "c3_bottom",
            420,
            250,
            570,
            270,
        ),
        block(
            "c1_bottom",
            30,
            250,
            180,
            270,
        ),
    ]

    ordered = LayoutEngine().assign_columns_and_reading_order(
        blocks,
        page_width=600,
    )

    assert [b.id for b in ordered] == [
        "c1_top",
        "c1_bottom",
        "c2_top",
        "c2_bottom",
        "c3_top",
        "c3_bottom",
    ]

    assert [b.column_id for b in ordered] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


def test_asymmetric_sidebar_and_main_content_remain_separate_regions():

    blocks = [
        block(
            "main_top",
            230,
            100,
            590,
            125,
        ),
        block(
            "sidebar_top",
            20,
            100,
            180,
            125,
        ),
        block(
            "main_bottom",
            230,
            220,
            590,
            245,
        ),
        block(
            "sidebar_bottom",
            20,
            220,
            180,
            245,
        ),
    ]

    ordered = LayoutEngine().assign_columns_and_reading_order(
        blocks,
        page_width=600,
    )

    assert [b.id for b in ordered] == [
        "sidebar_top",
        "sidebar_bottom",
        "main_top",
        "main_bottom",
    ]

    assert [b.column_id for b in ordered] == [
        0,
        0,
        1,
        1,
    ]


def test_full_width_transition_is_emitted_between_column_regions():

    blocks = [
        block(
            "header",
            20,
            40,
            580,
            70,
        ),
        block(
            "left_top",
            20,
            100,
            260,
            125,
        ),
        block(
            "right_top",
            340,
            100,
            580,
            125,
        ),
        block(
            "section_heading",
            20,
            300,
            580,
            330,
        ),
        block(
            "left_after",
            20,
            360,
            260,
            385,
        ),
        block(
            "right_after",
            340,
            360,
            580,
            385,
        ),
    ]

    ordered = LayoutEngine().assign_columns_and_reading_order(
        blocks,
        page_width=600,
    )

    assert [b.id for b in ordered] == [
        "header",
        "left_top",
        "right_top",
        "section_heading",
        "left_after",
        "right_after",
    ]


def test_multi_page_order_is_page_stable():

    blocks = [
        block(
            "p2_right",
            340,
            100,
            580,
            125,
            page=2,
        ),
        block(
            "p1_left",
            20,
            100,
            260,
            125,
            page=1,
        ),
        block(
            "p2_left",
            20,
            100,
            260,
            125,
            page=2,
        ),
        block(
            "p1_right",
            340,
            100,
            580,
            125,
            page=1,
        ),
    ]

    ordered = LayoutEngine().assign_columns_and_reading_order(
        blocks,
        page_width=600,
    )

    assert [b.id for b in ordered] == [
        "p1_left",
        "p1_right",
        "p2_left",
        "p2_right",
    ]

    assert [
        b.reading_order
        for b in ordered
    ] == [0, 1, 2, 3]