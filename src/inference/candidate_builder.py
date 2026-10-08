from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.inference.anchors import AnchorCandidate, AnchorDetector

from src.inference.candidates import (
    CandidateStatus,
    ExperienceCandidate,
)

from src.inference.evidence_models import (
    AnchorEvidence,
    AnchorStrength,
    AnchorType,
    BlockEvidence,
    EvidenceSignal,
    EvidenceType,
)


class CandidateBuilder:
    """
    Build ExperienceCandidate objects from detected anchors.

    This component performs candidate construction only.

    It does NOT:
        - perform final segmentation
        - validate final metadata
        - resolve competing candidates
        - create ExperienceEpisode objects
        - calculate final production confidence
    """

    def __init__(self) -> None:
        self._counter = 0

    # ------------------------------------------------------------------
    # Candidate ID
    # ------------------------------------------------------------------

    def _next_candidate_id(self) -> str:
        """
        Generate a deterministic candidate identifier.

        Example:
            experience_candidate_001
        """

        self._counter += 1

        return (
            f"experience_candidate_"
            f"{self._counter:03d}"
        )

    # ------------------------------------------------------------------
    # Anchor → AnchorEvidence
    # ------------------------------------------------------------------

    def build_anchor_evidence(
        self,
        anchor: AnchorCandidate,
    ) -> AnchorEvidence:
        """
        Convert an AnchorCandidate into AnchorEvidence.

        The detector provides field-level observations.
        This method converts those observations into structured
        evidence without making a final experience decision.
        """

        anchor_type = self._infer_anchor_type(anchor)

        strength = self._infer_strength(
            anchor_type=anchor_type,
            anchor=anchor,
        )

        score = self._calculate_anchor_score(anchor)

        evidence = AnchorEvidence(
            anchor_type=anchor_type,
            strength=strength,
            score=score,
        )

        if anchor.block_id:
            evidence.add_block(anchor.block_id)

        # --------------------------------------------------------------
        # ROLE
        # --------------------------------------------------------------

        if anchor.role:
            evidence.add_supporting_evidence(
                EvidenceSignal(
                    signal_type=EvidenceType.ROLE,
                    value=anchor.role,
                    confidence=anchor.role_confidence,
                    source_block_id=anchor.block_id,
                    source_field="role",
                    explanation=(
                        "Role detected by AnchorDetector."
                    ),
                )
            )

        # --------------------------------------------------------------
        # COMPANY
        # --------------------------------------------------------------

        if anchor.company:
            evidence.add_supporting_evidence(
                EvidenceSignal(
                    signal_type=EvidenceType.COMPANY,
                    value=anchor.company,
                    confidence=anchor.company_confidence,
                    source_block_id=anchor.block_id,
                    source_field="company",
                    explanation=(
                        "Company detected by AnchorDetector."
                    ),
                )
            )

        # --------------------------------------------------------------
        # DATE
        # --------------------------------------------------------------

        if anchor.start_date or anchor.end_date:
            evidence.add_supporting_evidence(
                EvidenceSignal(
                    signal_type=EvidenceType.DATE,
                    value={
                        "start_date": anchor.start_date,
                        "end_date": anchor.end_date,
                        "is_current": anchor.is_current,
                    },
                    confidence=anchor.date_confidence,
                    source_block_id=anchor.block_id,
                    source_field="date",
                    explanation=(
                        "Date range detected by AnchorDetector."
                    ),
                )
            )

        # --------------------------------------------------------------
        # LOCATION
        # --------------------------------------------------------------

        if anchor.location:
            evidence.add_supporting_evidence(
                EvidenceSignal(
                    signal_type=EvidenceType.LOCATION,
                    value=anchor.location,
                    confidence=0.80,
                    source_block_id=anchor.block_id,
                    source_field="location",
                    explanation=(
                        "Location detected by AnchorDetector."
                    ),
                )
            )

        evidence.explanation = (
            self._build_anchor_explanation(anchor, anchor_type)
        )

        return evidence

    # ------------------------------------------------------------------
    # Anchor type
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_anchor_type(
        anchor: AnchorCandidate,
    ) -> AnchorType:
        """
        Determine the structural anchor pattern.

        This is classification of available evidence,
        not final experience validation.
        """

        has_role = bool(anchor.role)
        has_company = bool(anchor.company)
        has_date = bool(
            anchor.start_date
            or anchor.end_date
            or anchor.is_current
        )
        has_location = bool(anchor.location)

        if (
            has_role
            and has_company
            and has_date
        ):
            return AnchorType.ROLE_COMPANY_DATE

        if has_role and has_date:
            return AnchorType.ROLE_DATE

        if has_role and has_company:
            return AnchorType.ROLE_COMPANY

        if (
            has_company
            and has_date
            and has_location
        ):
            return AnchorType.COMPANY_DATE_LOCATION

        if (
            has_role
            and has_location
            and has_date
        ):
            return AnchorType.ROLE_LOCATION_DATE

        if has_role:
            return AnchorType.ROLE_ONLY

        if has_company:
            return AnchorType.COMPANY_ONLY

        return AnchorType.UNKNOWN

    # ------------------------------------------------------------------
    # Anchor strength
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_strength(
        anchor_type: AnchorType,
        anchor: AnchorCandidate,
    ) -> AnchorStrength:
        """
        Assign qualitative anchor strength.

        This is intentionally conservative.
        """

        if anchor_type in {
            AnchorType.ROLE_COMPANY_DATE,
            AnchorType.COMPANY_DATE_LOCATION,
            AnchorType.ROLE_LOCATION_DATE,
        }:
            return AnchorStrength.STRONG

        if anchor_type in {
            AnchorType.ROLE_DATE,
            AnchorType.ROLE_COMPANY,
        }:
            return AnchorStrength.MEDIUM

        if anchor_type in {
            AnchorType.ROLE_ONLY,
            AnchorType.COMPANY_ONLY,
        }:
            return AnchorStrength.WEAK

        return AnchorStrength.WEAK

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_anchor_score(
        anchor: AnchorCandidate,
    ) -> float:
        """
        Calculate a preliminary anchor score.

        This is NOT the final experience confidence.

        The score combines the detector's field confidences while
        remaining conservative when fields are missing.
        """

        scores: List[float] = []

        if anchor.role:
            scores.append(anchor.role_confidence)

        if anchor.company:
            scores.append(anchor.company_confidence)

        if anchor.start_date or anchor.end_date:
            scores.append(anchor.date_confidence)

        if anchor.location:
            scores.append(0.80)

        if not scores:
            return 0.0

        # Average available evidence.
        score = sum(scores) / len(scores)

        return round(
            max(0.0, min(1.0, score)),
            4,
        )

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_anchor_explanation(
        anchor: AnchorCandidate,
        anchor_type: AnchorType,
    ) -> str:
        """
        Produce a human-readable explanation for debugging
        and regression analysis.
        """

        parts: List[str] = []

        if anchor.role:
            parts.append("role")

        if anchor.company:
            parts.append("company")

        if anchor.start_date or anchor.end_date:
            parts.append("date")

        if anchor.location:
            parts.append("location")

        if not parts:
            return "No strong anchor fields detected."

        return (
            "Anchor constructed from "
            + ", ".join(parts)
            + f" ({anchor_type.value})."
        )

    # ------------------------------------------------------------------
    # Build candidate
    # ------------------------------------------------------------------

    def build_candidate(
        self,
        anchor: AnchorCandidate,
        block_ids: Optional[Iterable[str]] = None,
    ) -> ExperienceCandidate:
        """
        Construct one ExperienceCandidate from an anchor.

        The candidate remains in CANDIDATE state.
        """

        candidate_id = self._next_candidate_id()

        candidate = ExperienceCandidate(
            candidate_id=candidate_id,
            role=anchor.role,
            company=anchor.company,
            location=anchor.location,
            start_date=anchor.start_date,
            end_date=anchor.end_date,
            is_current=anchor.is_current,
            confidence=0.0,
            status=CandidateStatus.CANDIDATE,
        )

        # --------------------------------------------------------------
        # Source blocks
        # --------------------------------------------------------------

        if block_ids:
            for block_id in block_ids:
                candidate.add_block(block_id)

        elif anchor.block_id:
            candidate.add_block(anchor.block_id)

        # --------------------------------------------------------------
        # Anchor evidence
        # --------------------------------------------------------------

        anchor_evidence = self.build_anchor_evidence(
            anchor
        )

        candidate.evidence.metadata[
            "anchor_type"
        ] = anchor_evidence.anchor_type.value

        candidate.evidence.metadata[
            "anchor_strength"
        ] = anchor_evidence.strength.value

        candidate.evidence.metadata[
            "anchor_score"
        ] = anchor_evidence.score

        candidate.evidence.metadata[
            "anchor_explanation"
        ] = anchor_evidence.explanation

        for signal in anchor_evidence.supporting_evidence:
            candidate.evidence.add_signal(signal)

        for signal in anchor_evidence.contradictory_evidence:
            candidate.evidence.add_signal(signal)

        # --------------------------------------------------------------
        # Preliminary candidate confidence
        #
        # This is deliberately the anchor score only.
        # Later evidence fusion / confidence modules may replace it.
        # --------------------------------------------------------------

        candidate.confidence = anchor_evidence.score

        return candidate

    # ------------------------------------------------------------------
    # Build multiple candidates
    # ------------------------------------------------------------------

    def build_candidates(
        self,
        anchors: Iterable[AnchorCandidate],
    ) -> List[ExperienceCandidate]:
        """
        Build candidates for a collection of detected anchors.

        No candidate merging is performed here.
        """

        candidates: List[ExperienceCandidate] = []

        for anchor in anchors:
            candidate = self.build_candidate(anchor)

            candidates.append(candidate)

        return candidates

class MultiBlockCandidateBuilder(CandidateBuilder):
    """
    Assemble experience candidates from multiple DocumentBlock evidence
    objects.

    This class intentionally remains below CandidateBuilder so the original
    single-anchor candidate construction API remains backward compatible.

    Responsibilities:
        - convert DocumentBlock objects into BlockEvidence
        - identify strong candidate heads
        - associate nearby company/date/location evidence
        - preserve source-block provenance
        - preserve description-block provenance
        - build ExperienceCandidate objects

    This class does NOT:
        - perform final experience segmentation
        - validate final episodes
        - create ExperienceEpisode objects
        - replace the production ExperienceExtractor
    """

    def __init__(
        self,
        anchor_detector: Optional[AnchorDetector] = None,
    ) -> None:
        super().__init__()
        self.anchor_detector = anchor_detector or AnchorDetector()

    # ------------------------------------------------------------------
    # Section helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_key(block: Any) -> str:
        """
        Normalize a block's section name.

        Missing section information is represented as an empty string so
        blocks without section metadata remain usable.
        """

        section = getattr(
            block,
            "section",
            None,
        )

        if section is None:
            return ""

        return section.strip().lower()

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_key(block: Any) -> Tuple[int, int, int]:
        """
        Produce a deterministic document-order key.

        Reading order is preferred. Page and block position are used as
        fallbacks when available.
        """

        return (
            int(
                getattr(
                    block,
                    "page",
                    1,
                )
                or 1
            ),
            int(
                getattr(
                    block,
                    "reading_order",
                    0,
                )
                or 0
            ),
            int(
                getattr(
                    block,
                    "column_id",
                    0,
                )
                or 0
            ),
        )

    # ------------------------------------------------------------------
    # Context compatibility
    # ------------------------------------------------------------------

    @classmethod
    def _compatible_context(
        cls,
        left: Any,
        right: Any,
    ) -> bool:
        """
        Determine whether two blocks belong to a compatible local context.

        Experience evidence should not be assembled across unrelated
        sections or clearly unrelated pages/columns.
        """

        left_section = cls._section_key(left)
        right_section = cls._section_key(right)

        if (
            left_section
            and right_section
            and left_section != right_section
        ):
            return False

        left_page = getattr(
            left,
            "page",
            None,
        )

        right_page = getattr(
            right,
            "page",
            None,
        )

        if (
            left_page is not None
            and right_page is not None
            and abs(left_page - right_page) > 1
        ):
            return False

        return True

    # ------------------------------------------------------------------
    # Proximity
    # ------------------------------------------------------------------

    @classmethod
    def _proximity_score(
        cls,
        left: Any,
        right: Any,
    ) -> float:
        """
        Calculate a lightweight structural proximity score.

        This score is used for local evidence association only. It is not
        final experience confidence.
        """

        if not cls._compatible_context(
            left,
            right,
        ):
            return 0.0

        left_order = int(
            getattr(
                left,
                "reading_order",
                0,
            )
            or 0
        )

        right_order = int(
            getattr(
                right,
                "reading_order",
                0,
            )
            or 0
        )

        distance = abs(
            left_order - right_order
        )

        if distance == 0:
            return 1.0

        if distance == 1:
            return 0.95

        if distance == 2:
            return 0.85

        if distance == 3:
            return 0.70

        if distance <= 5:
            return 0.50

        return 0.20

    def _fuse_role_fragments(
        self,
        window: List[BlockEvidence],
    ) -> Optional[BlockEvidence]:
        """Return the primary role evidence, optionally fused with adjacent
        role-fragment evidence.

        Role continuation is structural evidence, not a new experience.
        Preserve the first role block as the provenance anchor while combining
        contiguous role fragments into one semantic role value.
        """
        role_items = [
            item
            for item in window
            if item.role
        ]

        if not role_items:
            return None

        primary = role_items[0]

        if len(role_items) == 1:
            return primary

        fused_parts = [primary.role.strip()]

        for item in role_items[1:]:
            if item.role and item.role.strip() not in fused_parts:
                fused_parts.append(item.role.strip())

        fused_role = " ".join(
            part
            for part in fused_parts
            if part
        ).strip()

        if not fused_role:
            return primary

        # Keep the original evidence object immutable in spirit:
        # create a copy with the fused semantic value.
        try:
            return primary.model_copy(
                update={
                    "role": fused_role,
                }
            )
        except AttributeError:
            return primary.copy(
                update={
                    "role": fused_role,
                }
            )

    # ------------------------------------------------------------------
    # Block evidence
    # ------------------------------------------------------------------

    def build_block_evidence(self, blocks):
        """
        Convert DocumentBlocks into provenance-preserving BlockEvidence.

        This method records observations only.

        It does NOT:
            - decide whether the block belongs to an experience
            - create an ExperienceEpisode
            - resolve competing entities
            - perform final metadata validation
        """

        from src.inference.evidence_models import (
            BlockEvidence,
            EvidenceSignal,
            EvidenceType,
            EvidencePolarity,
        )

        evidence = []

        for block in blocks:

            anchor = self.anchor_detector.analyze_document_block(
                block
            )

            signals = []

            # --------------------------------------------------------------
            # ROLE EVIDENCE
            # --------------------------------------------------------------

            if anchor.role:
                signals.append(
                    EvidenceSignal(
                        signal_type=EvidenceType.ROLE,
                        value=anchor.role,
                        confidence=anchor.role_confidence,
                        source_block_id=block.id,
                        source_field="role",
                        explanation=(
                            "Role observation produced by "
                            "AnchorDetector."
                        ),
                    )
                )

            # --------------------------------------------------------------
            # COMPANY EVIDENCE
            # --------------------------------------------------------------

            if anchor.company:
                signals.append(
                    EvidenceSignal(
                        signal_type=EvidenceType.COMPANY,
                        value=anchor.company,
                        confidence=anchor.company_confidence,
                        source_block_id=block.id,
                        source_field="company",
                        explanation=(
                            "Company observation produced by "
                            "AnchorDetector."
                        ),
                    )
                )

            # --------------------------------------------------------------
            # DATE EVIDENCE
            # --------------------------------------------------------------

            if anchor.start_date or anchor.end_date:
                signals.append(
                    EvidenceSignal(
                        signal_type=EvidenceType.DATE,
                        value={
                            "start_date": anchor.start_date,
                            "end_date": anchor.end_date,
                            "is_current": anchor.is_current,
                        },
                        confidence=anchor.date_confidence,
                        source_block_id=block.id,
                        source_field="date",
                        explanation=(
                            "Date observation produced by "
                            "AnchorDetector."
                        ),
                    )
                )

            # --------------------------------------------------------------
            # LOCATION EVIDENCE
            # --------------------------------------------------------------

            if anchor.location:
                signals.append(
                    EvidenceSignal(
                        signal_type=EvidenceType.LOCATION,
                        value=anchor.location,
                        confidence=0.80,
                        source_block_id=block.id,
                        source_field="location",
                        explanation=(
                            "Location observation produced by "
                            "AnchorDetector."
                        ),
                    )
                )

            # --------------------------------------------------------------
            # STRUCTURAL NEGATIVE EVIDENCE
            # --------------------------------------------------------------
            #
            # A list/bullet block is useful evidence that the block is
            # probably descriptive content rather than an organization or
            # role header.
            #
            # This does NOT reject the block.
            # It simply records contradictory evidence for downstream
            # fusion.
            # --------------------------------------------------------------

            if block.is_list_item:
                signals.append(
                    EvidenceSignal(
                        signal_type=EvidenceType.STRUCTURAL,
                        value="list_item",
                        confidence=block.confidence,
                        polarity=EvidencePolarity.CONTRADICT,
                        source_block_id=block.id,
                        source_field="anchor",
                        explanation=(
                            "Block is explicitly marked as a "
                            "list item; this is contradictory evidence "
                            "against treating it as an experience header."
                        ),
                    )
                )

            # --------------------------------------------------------------
            # SENTENCE-LIKE NEGATIVE EVIDENCE
            # --------------------------------------------------------------

            sentence_like = (
                self.anchor_detector._looks_like_sentence(
                    block.text.strip()
                )
            )

            if sentence_like:
                signals.append(
                    EvidenceSignal(
                        signal_type=EvidenceType.SEMANTIC,
                        value="sentence_like",
                        confidence=block.confidence,
                        polarity=EvidencePolarity.CONTRADICT,
                        source_block_id=block.id,
                        source_field="anchor",
                        explanation=(
                            "Block is sentence-like; this is "
                            "contradictory evidence against treating "
                            "the block as a compact experience header."
                        ),
                    )
                )

            # --------------------------------------------------------------
            # EXPLICIT SOURCE / LAYOUT METADATA
            # --------------------------------------------------------------

            metadata = dict(
                getattr(block, "metadata", {}) or {}
            )

            raw_text = metadata.get("raw_text")

            normalized_text = metadata.get(
                "normalized_text"
            )

            layout_region = metadata.get(
                "layout_region"
            )

            section_confidence = metadata.get(
                "section_confidence",
                0.0,
            )

            # Defensive normalization of externally supplied metadata.
            if not isinstance(section_confidence, (int, float)):
                section_confidence = 0.0

            section_confidence = max(
                0.0,
                min(1.0, float(section_confidence)),
            )

            bbox = {
                "x0": block.bbox.x0,
                "y0": block.bbox.y0,
                "x1": block.bbox.x1,
                "y1": block.bbox.y1,
            }

            # --------------------------------------------------------------
            # BLOCK EVIDENCE
            # --------------------------------------------------------------

            block_evidence = BlockEvidence(
                block_id=block.id,
                text=block.text,

                raw_text=raw_text,
                normalized_text=normalized_text,

                section=block.section,
                section_confidence=section_confidence,

                page=block.page,
                column_id=block.column_id,
                reading_order=block.reading_order,
                bbox=bbox,

                layout_region=layout_region,

                is_bold=block.is_bold,
                is_heading=block.is_heading,

                block_type=block.block_type,
                semantic_label=block.semantic_label,

                source_type=block.source_type,
                extraction_method=block.extraction_method,

                detector="AnchorDetector",
                source_confidence=block.confidence,

                provenance={
                    "document_block_id": block.id,
                    "source_type": block.source_type,
                    "extraction_method": block.extraction_method,
                },

                role=anchor.role,
                company=anchor.company,

                start_date=anchor.start_date,
                end_date=anchor.end_date,
                is_current=anchor.is_current,

                location=anchor.location,

                role_confidence=anchor.role_confidence,
                company_confidence=anchor.company_confidence,
                date_confidence=anchor.date_confidence,
                location_confidence=(
                    0.80
                    if anchor.location
                    else 0.0
                ),

                signals=signals,

                metadata={
                    **metadata,
                    "is_anchor_head": anchor.is_anchor_head,
                    "is_sentence_like": sentence_like,
                    "is_list_item": block.is_list_item,
                    "is_bold": block.is_bold,
                },
            )

            evidence.append(block_evidence)

        return evidence

    # ------------------------------------------------------------------
    # Anchor classification
    # ------------------------------------------------------------------

    @staticmethod
    def _strong_role_anchor(
        evidence: BlockEvidence,
    ) -> bool:
        """
        Determine whether evidence is strong enough to begin an experience
        candidate.

        Role continuation blocks do not form independent experience
        boundaries.
        """

        if evidence.metadata.get(
            "is_role_continuation_block",
            False,
        ):
            return False

        if not evidence.has_role:
            return False

        if evidence.has_company and evidence.has_date:
            return True

        if evidence.has_date:
            return True

        if (
            evidence.role_confidence >= 0.70
            and not evidence.metadata.get(
                "is_sentence_like",
                False,
            )
        ):
            return True

        return False

    @staticmethod
    def _strong_non_role_anchor(
        evidence: BlockEvidence,
    ) -> bool:
        """
        Identify non-role metadata blocks that can support a nearby role.
        """

        return bool(
            evidence.has_company
            or evidence.has_date
            or evidence.has_location
        )

    # ------------------------------------------------------------------
    # Supporting evidence
    # ------------------------------------------------------------------

    @classmethod
    def _has_local_supporting_evidence(
        cls,
        evidence: List[BlockEvidence],
        index: int,
    ) -> bool:
        """
        Check whether a role has nearby company/date/location evidence.
        """

        current = evidence[index]

        start = max(
            0,
            index - 2,
        )

        end = min(
            len(evidence),
            index + 3,
        )

        for candidate_index in range(
            start,
            end,
        ):
            if candidate_index == index:
                continue

            candidate = evidence[
                candidate_index
            ]

            if not cls._compatible_context(
                current,
                candidate,
            ):
                continue

            if cls._strong_non_role_anchor(
                candidate
            ):
                return True

        return False

    def _is_role_continuation_block(
        self,
        evidence: List[BlockEvidence],
        index: int,
    ) -> bool:
        """
        Determine whether the current role observation is a continuation
        of the immediately preceding role block.

        This is intentionally conservative:
        - the previous block must contain a role
        - the previous role must be recognized as a role/domain continuation
        - the current block must contain a role
        - neither block may already contain company/date evidence
        - both blocks must belong to the same local context
        """

        if index <= 0:
            return False

        current = evidence[index]
        previous = evidence[index - 1]

        if not current.has_role or not previous.has_role:
            return False

        if (
            current.has_company
            or current.has_date
            or previous.has_company
            or previous.has_date
        ):
            return False

        if not self._compatible_context(current, previous):
            return False

        previous_role = (previous.role or "").strip()

        if not previous_role:
            return False

        # Reuse the canonical detector's existing continuation logic.
        if not self.anchor_detector._is_role_continuation(previous_role):
            return False

        return True

    # ------------------------------------------------------------------
    # Candidate head
    # ------------------------------------------------------------------

    def _is_candidate_head(
        self,
        evidence: List[BlockEvidence],
        index: int,
    ) -> bool:
        """
        Determine whether a BlockEvidence item can begin a candidate.

        A role may span multiple adjacent document blocks. A role
        continuation block must not start a new experience candidate.
        """

        item = evidence[index]

        # --------------------------------------------------------------
        # 1. Complete role + company + date is a strong anchor
        # --------------------------------------------------------------

        if (
            item.has_role
            and item.has_company
            and item.has_date
        ):
            item.metadata[
                "is_anchor_head"
            ] = True

            return True

        # --------------------------------------------------------------
        # 2. Do not start a new candidate from a role continuation block
        # --------------------------------------------------------------

        if self._is_role_continuation_block(
            evidence,
            index,
        ):
            item.metadata[
                "is_role_continuation_block"
            ] = True

            return False

        # --------------------------------------------------------------
        # 3. Existing role-anchor logic
        # --------------------------------------------------------------

        if (
            item.has_role
            and not item.metadata.get(
                "is_sentence_like",
                False,
            )
        ):
            if self._strong_role_anchor(item):
                item.metadata[
                    "is_anchor_head"
                ] = True

                return True

        # --------------------------------------------------------------
        # 4. Role with local supporting evidence
        # --------------------------------------------------------------

        if (
            item.has_role
            and self._has_local_supporting_evidence(
                evidence,
                index,
            )
        ):
            item.metadata[
                "is_anchor_head"
            ] = True

            return True

        # ----------------------------------------------------------
        # NON-ROLE EXPERIENCE CANDIDATE
        # ----------------------------------------------------------
        # A legitimate experience may have missing role metadata.
        # Preserve company/date evidence as a candidate instead of
        # discarding it. Validation will determine whether the
        # incomplete candidate is acceptable.
        if self._strong_non_role_anchor(item):
            if self._has_local_supporting_evidence(
                evidence,
                index,
            ):
                # ------------------------------------------------------
                # A non-role anchor must not start a separate candidate
                # when compatible role evidence follows within the local
                # candidate context.
                #
                # Supported layouts:
                #
                #   COMPANY
                #   ROLE
                #
                #   COMPANY
                #   DATE
                #   ROLE
                #
                #   DATE
                #   ROLE
                #
                # In these layouts the non-role evidence belongs to the
                # following role candidate.
                # ------------------------------------------------------

                lookahead_limit = min(
                    len(evidence),
                    index + 4,
                )

                for lookahead_index in range(
                    index + 1,
                    lookahead_limit,
                ):
                    lookahead = evidence[lookahead_index]

                    # A strong role appearing locally means the current
                    # non-role evidence should be associated with that
                    # role rather than start a separate candidate.
                    if lookahead.has_role:
                        item.metadata[
                            "is_role_association_anchor"
                        ] = True
                        return False

                    # A new strong candidate boundary means we should not
                    # look beyond it for role association.
                    if (
                        lookahead.has_company
                        and lookahead_index > index + 1
                    ):
                        break

                item.metadata["is_anchor_head"] = True
                return True

        return False

    # ------------------------------------------------------------------
    # Candidate window
    # ------------------------------------------------------------------

    def _candidate_window(
        self,
        evidence: List[BlockEvidence],
        head_index: int,
    ) -> List[BlockEvidence]:
        """
        Build the local evidence window for one experience candidate.

        Supports both:

            ROLE -> COMPANY -> DATE -> DESCRIPTION

        and:

            COMPANY -> DATE -> ROLE -> DESCRIPTION

        A previous strong role anchor is treated as a hard
        experience boundary.
        """

        if not evidence:
            return []

        head = evidence[head_index]

        window: List[BlockEvidence] = [head]

        # ---------------------------------------------------------
        # 1. Controlled backward scan
        #
        # Only nearby metadata is allowed to be inherited from
        # before the role.
        #
        # A previous strong role anchor is a hard boundary.
        # ---------------------------------------------------------
        backward_items: List[BlockEvidence] = []

        for index in range(head_index - 1, -1, -1):
            item = evidence[index]

            # -----------------------------------------------------
            # HARD EXPERIENCE BOUNDARY
            #
            # Never allow metadata from a previous role to leak into
            # the current candidate.
            # -----------------------------------------------------
            if self._strong_role_anchor(item):
                backward_items.clear()
                break

            # -----------------------------------------------------
            # Context compatibility
            #
            # Do not inherit metadata from an unrelated block merely
            # because it happens to contain a company/date/location.
            # -----------------------------------------------------
            if not self._compatible_context(head, item):
                break

            # -----------------------------------------------------
            # Only metadata-like evidence may be associated
            # backwards with the current role.
            # -----------------------------------------------------
            if (
                item.company
                or item.start_date
                or item.end_date
                or item.location
            ):
                backward_items.append(item)
                continue

            # Do not absorb arbitrary description blocks from the
            # previous experience.
            break

        # Restore original reading order.
        backward_items.reverse()

        # ---------------------------------------------------------
        # 2. Forward scan
        #
        # ROLE -> COMPANY -> DATE -> DESCRIPTION
        # ---------------------------------------------------------
        forward_items: List[BlockEvidence] = []

        for index in range(head_index + 1, len(evidence)):
            item = evidence[index]

            # ----------------------------------------------------------
            # Role continuation blocks belong to the current candidate.
            # They must not create a new experience boundary.
            # ----------------------------------------------------------

            if self._is_role_continuation_block(
                evidence,
                index,
            ):
                forward_items.append(item)
                continue

            if self._strong_role_anchor(item):
                backward_items.clear()
                break

            # Stop when the next block is no longer compatible with
            # the current candidate's local context.
            if not self._compatible_context(head, item):
                break

            # Require local supporting evidence for additional blocks.
            if not self._has_local_supporting_evidence(
                evidence,
                index,
            ):
                if self._proximity_score(head, item) <= 0:
                    break

            forward_items.append(item)

        # ---------------------------------------------------------
        # 3. Combine
        #
        # Preserve original document reading order:
        #
        # backward metadata -> role head -> forward evidence
        # ---------------------------------------------------------
        return backward_items + window + forward_items


    # ------------------------------------------------------------------
    # Build from blocks
    # ------------------------------------------------------------------

    def build_from_blocks(
        self,
        blocks: Iterable[Any],
    ) -> List[ExperienceCandidate]:

        blocks = [
            block
            for block in blocks
            if (
                not getattr(
                    block,
                    "section",
                    None,
                )
                or block.section.strip().lower()
                == "experience"
            )
        ]

        evidence = self.build_block_evidence(
            blocks
        )

        candidates: List[
            ExperienceCandidate
        ] = []

        consumed: set[int] = set()

        for index, item in enumerate(
            evidence
        ):
            if index in consumed:
                continue

            if not self._is_candidate_head(
                evidence,
                index,
            ):
                continue

            window = (
                self._candidate_window(
                    evidence,
                    index,
                )
            )

            if not window:
                continue

            role_item = self._fuse_role_fragments(window)

            company_items = [
                candidate
                for candidate in window
                if candidate.company
            ]

            anchor_reading_order = (
                role_item.reading_order
                if role_item
                else window[0].reading_order
            )

            company_item = (
                min(
                    company_items,
                    key=lambda candidate: (
                        abs(
                            candidate.reading_order
                            - anchor_reading_order
                        ),
                        candidate.reading_order,
                    ),
                )
                if company_items
                else None
            )

            date_item = next(
                (
                    candidate
                    for candidate in window
                    if candidate.has_date
                ),
                None,
            )

            location_item = next(
                (
                    candidate
                    for candidate in window
                    if candidate.location
                ),
                None,
            )

            candidate = ExperienceCandidate(
                candidate_id=(
                    self._next_candidate_id()
                ),
                role=role_item.role if role_item else None,
                company=(
                    company_item.company
                    if company_item
                    else None
                ),
                location=(
                    location_item.location
                    if location_item
                    else None
                ),
                start_date=(
                    date_item.start_date
                    if date_item
                    else None
                ),
                end_date=(
                    date_item.end_date
                    if date_item
                    else None
                ),
                is_current=(
                    date_item.is_current
                    if date_item
                    else False
                ),
                confidence=0.0,
                status=CandidateStatus.CANDIDATE,
            )

            block_ids = [
                item.block_id
                for item in window
            ]

            for block_id in block_ids:
                candidate.add_block(
                    block_id
                )

            # ----------------------------------------------------------
            # Provenance
            # ----------------------------------------------------------

            field_sources: Dict[str, str] = {}

            if role_item:
                field_sources["role"] = role_item.block_id

            if company_item:
                field_sources["company"] = company_item.block_id

            if date_item:
                field_sources["date"] = date_item.block_id

            if location_item:
                field_sources["location"] = location_item.block_id

            if company_item:
                field_sources[
                    "company"
                ] = company_item.block_id

            if date_item:
                field_sources[
                    "date"
                ] = date_item.block_id

            if location_item:
                field_sources[
                    "location"
                ] = location_item.block_id

            candidate.evidence.metadata[
                "assembly"
            ] = "multi_block"

            candidate.evidence.metadata[
                "field_sources"
            ] = field_sources

            candidate.evidence.metadata[
                "source_blocks"
            ] = block_ids

            candidate.evidence.metadata[
                "candidate_head"
            ] = (
                role_item.block_id
                if role_item
                else window[0].block_id
            )

            # ----------------------------------------------------------
            # Add field-level evidence
            # ----------------------------------------------------------

            if role_item and role_item.role:
                candidate.evidence.add_signal(
                    EvidenceSignal(
                        signal_type=EvidenceType.ROLE,
                        value=role_item.role,
                        confidence=role_item.role_confidence,
                        source_block_id=(
                            role_item.block_id
                        ),
                        source_field="role",
                        explanation=(
                            "Role detected from "
                            "experience block evidence."
                        ),
                    )
                )

            if company_item and company_item.company:
                candidate.evidence.add_signal(
                    EvidenceSignal(
                        signal_type=EvidenceType.COMPANY,
                        value=company_item.company,
                        confidence=company_item.company_confidence,
                        source_block_id=(
                            company_item.block_id
                        ),
                        source_field="company",
                        explanation=(
                            "Company associated "
                            "from local evidence."
                        ),
                    )
                )

            if date_item:
                candidate.evidence.add_signal(
                    EvidenceSignal(
                        signal_type=EvidenceType.DATE,
                        value={
                            "start_date": (
                                date_item.start_date
                            ),
                            "end_date": (
                                date_item.end_date
                            ),
                            "is_current": (
                                date_item.is_current
                            ),
                        },
                        confidence=(
                            date_item.date_confidence
                        ),
                        source_block_id=(
                            date_item.block_id
                        ),
                        source_field="date",
                        explanation=(
                            "Date associated from "
                            "local experience evidence."
                        ),
                    )
                )

            if (
                location_item
                and location_item.location
            ):
                candidate.evidence.add_signal(
                    EvidenceSignal(
                        signal_type=EvidenceType.LOCATION,
                        value=location_item.location,
                        confidence=(
                            location_item.location_confidence
                        ),
                        source_block_id=(
                            location_item.block_id
                        ),
                        source_field="location",
                        explanation=(
                            "Location associated from "
                            "local experience evidence."
                        ),
                    )
                )

            # ----------------------------------------------------------
            # Description provenance
            # ----------------------------------------------------------

            description_blocks: List[str] = []

            role_block_id = (
                role_item.block_id
                if role_item
                else None
            )

            for window_item in window:
                if (
                    role_block_id
                    and window_item.block_id == role_block_id
                ):
                    continue

                if (
                    company_item
                    and window_item.block_id
                    == company_item.block_id
                ):
                    continue

                if (
                    date_item
                    and window_item.block_id
                    == date_item.block_id
                ):
                    continue

                if (
                    location_item
                    and window_item.block_id
                    == location_item.block_id
                ):
                    continue

                text = (
                    window_item.text.strip()
                )

                if not text:
                    continue

                if (
                    window_item.role
                    or window_item.company
                    or window_item.has_date
                ):
                    continue

                description_blocks.append(
                    window_item.block_id
                )

            candidate.description_block_ids = (
                description_blocks
            )

            # ----------------------------------------------------------
            # Preliminary assembled confidence
            # ----------------------------------------------------------

            candidate.confidence = (
                self._assembled_confidence(
                    role_item=role_item,
                    company_item=company_item,
                    date_item=date_item,
                    location_item=location_item,
                )
            )

            candidate.evidence.metadata[
                "description_blocks"
            ] = description_blocks

            candidates.append(
                candidate
            )

            # Consume only blocks that were actually assembled into this
            # candidate. This prevents a later role from stealing the same
            # company/date evidence while allowing independent roles to
            # remain separate.
            for window_item in window:
                try:
                    evidence_index = (
                        evidence.index(
                            window_item
                        )
                    )

                    consumed.add(
                        evidence_index
                    )
                except ValueError:
                    continue

        return candidates

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _assembled_confidence(
        *,
        role_item: Optional[BlockEvidence],
        company_item: Optional[BlockEvidence],
        date_item: Optional[BlockEvidence],
        location_item: Optional[BlockEvidence],
    ) -> float:
        """
        Calculate preliminary assembled confidence.

        This is intentionally NOT final production confidence.
        """

        scores: List[float] = []

        if role_item and role_item.role:
            scores.append(role_item.role_confidence)

        if company_item and company_item.company:
            scores.append(company_item.company_confidence)

        if date_item:
            scores.append(date_item.date_confidence)

        if location_item and location_item.location:
            scores.append(location_item.location_confidence)

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 2)  