"""Production-oriented work/research/internship experience extraction."""

from typing import List, Optional

from src.document_models import (
    DocumentBlock,
    ExperienceEpisode,
)

from src.inference.segmenter import (
    ExperienceSegmenter,
)

from src.inference.confidence import (
    ConfidenceScorer,
)

from src.extraction.skills import (
    SkillExtractor,
)

from src.inference.metadata import MetadataResolver

from src.inference.candidate_builder import (
    MultiBlockCandidateBuilder,
)
from src.inference.experience_pipeline import (
    ExperienceEpisodeMaterializer,
)
from src.inference.validation import (
    ExperienceCandidateValidator,
)

class ExperienceExtractor:

    def __init__(
        self,
        skill_extractor: Optional[
            SkillExtractor
        ] = None,
    ):

        self.segmenter = (
            ExperienceSegmenter()
        )

        self.confidence_scorer = (
            ConfidenceScorer()
        )

        self.metadata_resolver = MetadataResolver()

        self.skill_extractor = (
            skill_extractor
            or SkillExtractor()
        )

        

    def extract(
        self,
        blocks: List[DocumentBlock],
    ) -> List[ExperienceEpisode]:
        """Extract discrete professional experience episodes.

        Primary source:
            Canonical ``experience`` section.

        Fallback:
            Evidence-based scanning for resumes that use unusual section
            headings or do not expose a standard experience heading.

        The fallback deliberately avoids treating every technical sentence
        as employment history.
        """

        # =========================================================
        # PRIMARY PATH
        # =========================================================

        experience_blocks = [
            block
            for block in blocks
            if block.section == "experience"
        ]

        # =========================================================
        # FALLBACK PATH
        # =========================================================

        if not experience_blocks:

            detector = (
                self.segmenter
                .anchor_detector
            )

            candidates = []

            for block in blocks:

                if block.is_heading:
                    continue

                # Do not reinterpret explicitly classified non-experience
                # sections as work experience.
                if block.section not in (None, "", "unknown"):
                    continue

                anchor = (
                    detector.analyze_block(
                        block.text,
                        is_bold=block.is_bold,
                    )
                )

                # Only accept blocks containing meaningful anchor evidence.
                if (
                    anchor.is_anchor_head
                    or anchor.start_date
                ):
                    candidates.append(block)

            experience_blocks = candidates

        if not experience_blocks:
            return []

        # =========================================================
        # SEGMENT
        # =========================================================

        episodes = (
            self.segmenter
            .segment_experiences(
                experience_blocks
            )
        )

        # =========================================================
        # ENRICH
        # =========================================================

        for episode in episodes:

            # ---------------------------------------------------------
            # Metadata normalization / resolution
            # ---------------------------------------------------------

            episode = (
                self.metadata_resolver
                .resolve_episode(episode)
            )

            # ---------------------------------------------------------
            # Skill / technology extraction
            # ---------------------------------------------------------

            combined_text = "\n".join(
                value
                for value in (
                    episode.role,
                    episode.company,
                    episode.location,
                    episode.description,
                    *episode.bullet_points,
                )
                if value
            )

            episode.technologies_mentioned = (
                self.skill_extractor
                .extract_from_text(
                    combined_text
                )
            )

            # ---------------------------------------------------------
            # Confidence scoring
            # ---------------------------------------------------------

            episode.confidence = (
                self.confidence_scorer
                .score_experience_episode(
                    episode
                )
            )

        return episodes

    def extract_evidence_based(
        self,
        blocks: List[DocumentBlock],
    ) -> List[ExperienceEpisode]:
        """
        Build experience episodes through the evidence-first inference path.

        This is a shadow/experimental production integration path.

        It does NOT replace ``extract()``.

        Pipeline:

            DocumentBlock
                ↓
            MultiBlockCandidateBuilder
                ↓
            ExperienceCandidate
                ↓
            ExperienceCandidateValidator
                ↓
            ExperienceEpisodeMaterializer
                ↓
            ExperienceEpisode

        The existing ``extract()`` method remains the production
        authoritative path until parity and real-resume validation
        establish a migration gate.
        """

        experience_blocks = [
            block
            for block in blocks
            if block.section == "experience"
        ]

        if not experience_blocks:
            return []

        candidate_builder = MultiBlockCandidateBuilder()

        candidates = candidate_builder.build_from_blocks(
            experience_blocks
        )

        validator = ExperienceCandidateValidator()

        materializer = ExperienceEpisodeMaterializer(
            validator=validator,
        )

        episodes: List[ExperienceEpisode] = []

        for candidate in candidates:

            validation = validator.validate(
                candidate
            )

            if not validation.is_valid:
                continue

            episode = materializer.materialize(
                candidate=candidate,
                validation=validation,
            )

            # Preserve the existing downstream enrichment contract.
            combined_text = "\n".join(
                value
                for value in (
                    episode.role,
                    episode.company,
                    episode.location,
                    episode.description,
                    *episode.bullet_points,
                )
                if value
            )

            episode.technologies_mentioned = (
                self.skill_extractor.extract_from_text(
                    combined_text
                )
            )

            episode.confidence = (
                self.confidence_scorer
                .score_experience_episode(
                    episode
                )
            )

            episodes.append(episode)

        return episodes