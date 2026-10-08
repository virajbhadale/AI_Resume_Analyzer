"""
Universal experience candidate tests.

These tests validate structural behavior rather than individual
real-resume names or layouts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.inference.anchors import AnchorDetector
from src.inference.candidate_builder import MultiBlockCandidateBuilder

from tests.fixtures.experience_cases import (
    UNIVERSAL_EXPERIENCE_CASES,
)

from src.document_models import DocumentBlock

def _make_blocks(case):
    blocks = []

    for index, text in enumerate(case.blocks):
        blocks.append(
            DocumentBlock(
                id=f"{case.case_id}_{index}",
                page=1,
                text=text,
                is_bold=False,
                is_heading=False,
                column_id=0,
                reading_order=index,
                source_type="synthetic",
                extraction_method="test_fixture",
                block_type="text",
                section="experience",
            )
        )

    return blocks


@pytest.mark.parametrize(
    "case",
    UNIVERSAL_EXPERIENCE_CASES,
    ids=lambda case: case.case_id,
)
def test_universal_experience_case(case):
    detector = AnchorDetector()

    builder = MultiBlockCandidateBuilder(
        anchor_detector=detector,
    )

    blocks = _make_blocks(case)

    candidates = builder.build_from_blocks(blocks)

    assert candidates, (
        f"No candidates produced for universal case "
        f"{case.case_id!r}"
    )

    assert len(candidates) == len(case.expected), (
        f"{case.case_id}: expected "
        f"{len(case.expected)} candidates, got "
        f"{len(candidates)}"
    )

    for candidate, expected in zip(
        candidates,
        case.expected,
    ):
        if expected.role is not None:
            assert candidate.role == expected.role

        if expected.company is not None:
            assert candidate.company == expected.company

        if expected.location is not None:
            assert candidate.location == expected.location

        if expected.start_date is not None:
            assert str(candidate.start_date) == expected.start_date

        if expected.end_date is not None:
            assert str(candidate.end_date) == expected.end_date

        if expected.is_current is not None:
            assert candidate.is_current == expected.is_current