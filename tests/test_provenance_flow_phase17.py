"""
Phase 17 — Provenance Flow Tests

Purpose
-------
Verify provenance across the confirmed candidate-generation path:

    DocumentBlock
        ↓
    AnchorDetector
        ↓
    BlockEvidence / EvidenceSignal
        ↓
    MultiBlockCandidateBuilder
        ↓
    ExperienceCandidate

These tests intentionally stop at candidate generation.

They do not assume an unconfirmed ExperienceSegmenter API.
"""

from __future__ import annotations

from typing import Any, Iterable, List

from src.document_models import BBox, DocumentBlock
from src.inference.anchors import AnchorDetector
from src.inference.candidate_builder import MultiBlockCandidateBuilder
from src.inference.evidence_models import EvidencePolarity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_block(
    block_id: str,
    text: str,
    reading_order: int,
    *,
    page: int = 1,
    column_id: int = 0,
    bold: bool = False,
    section: str = "experience",
    y: float = 100.0,
    list_item: bool = False,
) -> DocumentBlock:
    """
    Create a canonical DocumentBlock using the repository's actual model.
    """

    return DocumentBlock(
        id=block_id,
        page=page,
        page_width=600.0,
        page_height=800.0,
        text=text,
        bbox=BBox(
            x0=40.0,
            y0=y,
            x1=560.0,
            y1=y + 20.0,
        ),
        reading_order=reading_order,
        column_id=column_id,
        is_bold=bold,
        is_heading=False,
        is_list_item=list_item,
        block_type="text",
        section=section,
    )


def build_candidates(
    blocks: List[DocumentBlock],
):
    """
    Run the confirmed candidate-generation path.

    The production builder consumes self.anchor_detector during
    build_block_evidence(), therefore inject the detector explicitly.
    """

    detector = AnchorDetector()

    builder = MultiBlockCandidateBuilder(
        anchor_detector=detector,
    )

    return builder.build_from_blocks(blocks)


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def flatten_candidate_evidence(
    candidate: Any,
) -> List[Any]:
    """
    Return EvidenceSignal objects from the candidate evidence collection.

    The current candidate evidence representation may contain tuples
    rather than exposing EvidenceSignal directly.

    Supported forms:

        EvidenceSignal
        (label, EvidenceSignal)
        (EvidenceSignal, label)

    Unknown tuple contents are ignored rather than guessed.
    """

    signals: List[Any] = []

    for item in candidate.evidence:
        if hasattr(item, "source_block_id"):
            signals.append(item)
            continue

        if isinstance(item, tuple):
            for value in item:
                if hasattr(value, "source_block_id"):
                    signals.append(value)
                    break

    return signals


# ---------------------------------------------------------------------------
# 1. Basic source-block provenance
# ---------------------------------------------------------------------------

def test_candidate_provenance_preserves_source_blocks():
    """
    A candidate assembled from multiple document blocks must retain
    its source block identifiers.
    """

    blocks = [
        make_block(
            "b1",
            "Machine Learning Engineer",
            1,
            bold=True,
            y=100,
        ),
        make_block(
            "b2",
            "OpenAI",
            2,
            y=125,
        ),
        make_block(
            "b3",
            "Jan 2024 - Present",
            3,
            y=150,
        ),
        make_block(
            "b4",
            "Built production inference systems.",
            4,
            y=175,
        ),
    ]

    candidates = build_candidates(blocks)

    assert candidates

    candidate = candidates[0]

    source_blocks = set(
        candidate.evidence.metadata.get(
            "source_blocks",
            [],
        )
    )

    assert {"b1", "b2", "b3"}.issubset(
        source_blocks
    )


# ---------------------------------------------------------------------------
# 2. Multi-block candidate provenance
# ---------------------------------------------------------------------------

def test_candidate_provenance_survives_candidate_assembly():
    """
    Role, company and date blocks contributing to one candidate must
    remain traceable after candidate construction.
    """

    blocks = [
        make_block(
            "b1",
            "Data Scientist",
            1,
            bold=True,
            y=100,
        ),
        make_block(
            "b2",
            "Acme AI Labs",
            2,
            y=125,
        ),
        make_block(
            "b3",
            "Jan 2022 - Dec 2023",
            3,
            y=150,
        ),
        make_block(
            "b4",
            "Developed machine learning models.",
            4,
            y=175,
        ),
    ]

    candidates = build_candidates(blocks)

    assert candidates

    candidate = candidates[0]

    source_blocks = set(
        candidate.evidence.metadata.get(
            "source_blocks",
            [],
        )
    )

    assert "b1" in source_blocks
    assert "b2" in source_blocks
    assert "b3" in source_blocks


# ---------------------------------------------------------------------------
# 3. EvidenceSignal source provenance
# ---------------------------------------------------------------------------

def test_evidence_signals_retain_source_block_provenance():
    """
    EvidenceSignal objects produced by build_block_evidence() must retain
    the DocumentBlock that produced them.

    This validates provenance at the canonical BlockEvidence layer,
    where the production implementation explicitly stores signals.
    """

    blocks = [
        make_block(
            "b1",
            "Machine Learning Engineer",
            1,
            bold=True,
            y=100,
        ),
        make_block(
            "b2",
            "OpenAI",
            2,
            y=125,
        ),
        make_block(
            "b3",
            "2024 - Present",
            3,
            y=150,
        ),
    ]

    builder = MultiBlockCandidateBuilder(
        anchor_detector=AnchorDetector(),
    )

    block_evidence = builder.build_block_evidence(
        blocks
    )

    assert block_evidence

    signals = [
        signal
        for evidence in block_evidence
        for signal in evidence.signals
    ]

    assert signals

    source_block_ids = {
        signal.source_block_id
        for signal in signals
        if signal.source_block_id is not None
    }

    assert source_block_ids

    assert source_block_ids.issubset(
        {"b1", "b2", "b3"}
    )


# ---------------------------------------------------------------------------
# 4. Role/company/date evidence remains distinguishable
# ---------------------------------------------------------------------------

def test_role_company_date_evidence_remains_traceable():
    """
    Role, company and date EvidenceSignals must retain their source
    block and source field.
    """

    blocks = [
        make_block(
            "role",
            "Machine Learning Engineer",
            1,
            bold=True,
            y=100,
        ),
        make_block(
            "company",
            "OpenAI",
            2,
            y=125,
        ),
        make_block(
            "date",
            "Jan 2024 - Present",
            3,
            y=150,
        ),
    ]

    builder = MultiBlockCandidateBuilder(
        anchor_detector=AnchorDetector(),
    )

    block_evidence = builder.build_block_evidence(
        blocks
    )

    assert block_evidence

    signals = [
        signal
        for evidence in block_evidence
        for signal in evidence.signals
    ]

    assert signals

    role_signals = [
        signal
        for signal in signals
        if signal.source_field == "role"
    ]

    company_signals = [
        signal
        for signal in signals
        if signal.source_field == "company"
    ]

    date_signals = [
        signal
        for signal in signals
        if signal.source_field == "date"
    ]

    assert role_signals
    assert company_signals
    assert date_signals

    assert all(
        signal.source_block_id == "role"
        for signal in role_signals
    )

    assert all(
        signal.source_block_id == "company"
        for signal in company_signals
    )

    assert all(
        signal.source_block_id == "date"
        for signal in date_signals
    )


# ---------------------------------------------------------------------------
# 5. Provenance is deterministic
# ---------------------------------------------------------------------------

def test_candidate_provenance_is_deterministic():
    """
    Identical input blocks must produce identical provenance.
    """

    blocks = [
        make_block(
            "b1",
            "Data Analyst",
            1,
            bold=True,
            y=100,
        ),
        make_block(
            "b2",
            "Acme Analytics",
            2,
            y=125,
        ),
        make_block(
            "b3",
            "2023 - Present",
            3,
            y=150,
        ),
    ]

    candidates_a = build_candidates(blocks)
    candidates_b = build_candidates(blocks)

    assert candidates_a
    assert candidates_b

    provenance_a = candidates_a[0].evidence.metadata.get(
        "source_blocks",
        [],
    )

    provenance_b = candidates_b[0].evidence.metadata.get(
        "source_blocks",
        [],
    )

    assert provenance_a == provenance_b


# ---------------------------------------------------------------------------
# 6. Separate candidate provenance
# ---------------------------------------------------------------------------

def test_separate_candidates_do_not_share_source_blocks():
    """
    Clearly separate experiences must not share source blocks.
    """

    blocks = [
        make_block(
            "b1",
            "Data Analyst",
            1,
            bold=True,
            y=100,
        ),
        make_block(
            "b2",
            "Company A",
            2,
            y=125,
        ),
        make_block(
            "b3",
            "2020 - 2022",
            3,
            y=150,
        ),
        make_block(
            "b4",
            "ML Engineer",
            4,
            bold=True,
            y=220,
        ),
        make_block(
            "b5",
            "Company B",
            5,
            y=245,
        ),
        make_block(
            "b6",
            "2023 - Present",
            6,
            y=270,
        ),
    ]

    candidates = build_candidates(blocks)

    assert len(candidates) >= 2

    first_source_blocks = set(
        candidates[0].evidence.metadata.get(
            "source_blocks",
            [],
        )
    )

    second_source_blocks = set(
        candidates[1].evidence.metadata.get(
            "source_blocks",
            [],
        )
    )

    assert first_source_blocks
    assert second_source_blocks

    assert first_source_blocks.isdisjoint(
        second_source_blocks
    )


# ---------------------------------------------------------------------------
# 7. Explicit structural negative evidence
# ---------------------------------------------------------------------------

def test_negative_evidence_is_preserved():
    """
    Explicitly marked list/bullet blocks must produce contradictory
    structural evidence.

    This test deliberately sets is_list_item=True because that is the
    confirmed production condition for structural negative evidence.
    """

    blocks = [
        make_block(
            "b1",
            "- Presented findings at weekly research meetings",
            1,
            y=100,
            list_item=True,
        ),
    ]

    builder = MultiBlockCandidateBuilder(
        anchor_detector=AnchorDetector(),
    )

    block_evidence = builder.build_block_evidence(
        blocks
    )

    assert block_evidence

    signals = block_evidence[0].signals

    contradictory = [
        signal
        for signal in signals
        if signal.polarity == EvidencePolarity.CONTRADICT
    ]

    assert contradictory


# ---------------------------------------------------------------------------
# 8. Sentence-like prose
# ---------------------------------------------------------------------------

def test_sentence_like_prose_is_not_promoted_to_entity_evidence():
    """
    Sentence-like prose should not be treated as a compact experience
    header.

    The assertion is deliberately based on the actual detector output
    rather than assuming that every prose sentence must produce a
    contradictory signal.
    """

    blocks = [
        make_block(
            "b1",
            "Developed scalable machine learning models for production.",
            1,
            y=100,
        ),
    ]

    builder = MultiBlockCandidateBuilder(
        anchor_detector=AnchorDetector(),
    )

    block_evidence = builder.build_block_evidence(
        blocks
    )

    assert block_evidence

    evidence = block_evidence[0]

    assert evidence.metadata.get(
        "is_sentence_like"
    ) is True

    signals = evidence.signals

    contradictory = [
        signal
        for signal in signals
        if signal.polarity == EvidencePolarity.CONTRADICT
    ]

    assert contradictory


# ---------------------------------------------------------------------------
# 9. Missing metadata remains missing
# ---------------------------------------------------------------------------

def test_missing_metadata_is_not_fabricated():
    """
    A role-only block must not cause company/date values to be invented.
    """

    blocks = [
        make_block(
            "b1",
            "Research Intern",
            1,
            bold=True,
            y=100,
        ),
    ]

    candidates = build_candidates(blocks)

    assert candidates

    candidate = candidates[0]

    assert candidate.role == "Research Intern"

    assert candidate.company is None
    assert candidate.start_date is None
    assert candidate.end_date is None


# ---------------------------------------------------------------------------
# 10. Every EvidenceSignal has valid provenance
# ---------------------------------------------------------------------------

def test_every_signal_points_to_a_real_source_block():
    """
    No EvidenceSignal should reference a block that was not part of
    the input document.
    """

    blocks = [
        make_block(
            "b1",
            "AI Engineer",
            1,
            bold=True,
            y=100,
        ),
        make_block(
            "b2",
            "OpenAI",
            2,
            y=125,
        ),
        make_block(
            "b3",
            "2024 - Present",
            3,
            y=150,
        ),
    ]

    candidates = build_candidates(blocks)

    assert candidates

    valid_block_ids = {
        block.id
        for block in blocks
    }

    for candidate in candidates:
        signals = flatten_candidate_evidence(
            candidate
        )

        for signal in signals:
            if signal.source_block_id is not None:
                assert (
                    signal.source_block_id
                    in valid_block_ids
                )