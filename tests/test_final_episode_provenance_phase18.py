from src.document_models import ExperienceEpisode
from src.inference.candidates import ExperienceCandidate
from src.inference.evidence_models import (
    EvidenceCollection,
    EvidencePolarity,
    EvidenceSignal,
    EvidenceType,
)
from src.inference.experience_pipeline import (
    ExperienceEpisodeMaterializer,
)


def make_candidate() -> ExperienceCandidate:
    role_signal = EvidenceSignal(
        signal_type=EvidenceType.ROLE,
        value="Machine Learning Engineer",
        confidence=0.95,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b1",
        source_field="role",
        explanation="Role anchor detected.",
    )

    company_signal = EvidenceSignal(
        signal_type=EvidenceType.COMPANY,
        value="OpenAI",
        confidence=0.94,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b2",
        source_field="company",
        explanation="Company evidence detected.",
    )

    date_signal = EvidenceSignal(
        signal_type=EvidenceType.DATE,
        value="2024 - Present",
        confidence=0.96,
        polarity=EvidencePolarity.SUPPORT,
        source_block_id="b3",
        source_field="start_date",
        explanation="Experience date range detected.",
    )

    return ExperienceCandidate(
        candidate_id="candidate-001",
        block_ids=[
            "b1",
            "b2",
            "b3",
            "b4",
        ],
        role="Machine Learning Engineer",
        company="OpenAI",
        start_date="2024",
        end_date="Present",
        is_current=True,
        location="San Francisco, CA",
        evidence=EvidenceCollection(
            signals=[
                role_signal,
                company_signal,
                date_signal,
            ]
        ),
        confidence=0.92,
        metadata={
            "description": "Built production machine learning systems.",
            "bullet_points": [
                "Built inference pipelines.",
                "Improved model serving reliability.",
            ],
            "technologies_mentioned": [
                "Python",
                "PyTorch",
            ],
        },
    )


def test_materializes_candidate_into_episode():
    candidate = make_candidate()

    episode = ExperienceEpisodeMaterializer().materialize(
        candidate
    )

    assert isinstance(
        episode,
        ExperienceEpisode,
    )

    assert episode.id == "candidate-001"
    assert episode.role == "Machine Learning Engineer"
    assert episode.company == "OpenAI"
    assert episode.location == "San Francisco, CA"
    assert episode.start_date == "2024"
    assert episode.end_date == "Present"
    assert episode.is_current is True


def test_candidate_provenance_reaches_final_episode():
    episode = ExperienceEpisodeMaterializer().materialize(
        make_candidate()
    )

    evidence = episode.evidence

    assert evidence["provenance_version"] == 1

    assert evidence["candidate_id"] == (
        "candidate-001"
    )

    assert evidence["source_block_ids"] == [
        "b1",
        "b2",
        "b3",
        "b4",
    ]

    assert len(
        evidence["supporting_evidence"]
    ) == 3


def test_field_level_provenance_is_preserved():
    episode = ExperienceEpisodeMaterializer().materialize(
        make_candidate()
    )

    field_evidence = (
        episode.evidence["field_evidence"]
    )

    assert len(
        field_evidence["role"]
    ) == 1

    assert (
        field_evidence["role"][0]["value"]
        == "Machine Learning Engineer"
    )

    assert (
        field_evidence["company"][0]["value"]
        == "OpenAI"
    )

    assert (
        field_evidence["start_date"][0]["value"]
        == "2024 - Present"
    )


def test_metadata_resolution_records_missing_fields():
    candidate = make_candidate()

    candidate.company = None
    candidate.location = None

    episode = ExperienceEpisodeMaterializer().materialize(
        candidate
    )

    assert episode.company == (
        "Company Not Specified"
    )

    resolution = (
        episode.evidence[
            "metadata_resolution"
        ]
    )

    assert resolution["company_resolved"] is False
    assert resolution["location_resolved"] is False


def test_missing_metadata_does_not_invent_or_relabel_evidence():
    candidate = make_candidate()

    candidate.company = None

    episode = ExperienceEpisodeMaterializer().materialize(
        candidate
    )

    supporting = (
        episode.evidence[
            "supporting_evidence"
        ]
    )

    company_values = [
        item.get("value")
        for item in supporting
        if item.get("source_field")
        == "company"
    ]

    # Existing source evidence must be preserved.
    assert company_values == ["OpenAI"]

    # But the final resolved metadata must correctly indicate
    # that the candidate did not resolve a company.
    resolution = (
        episode.evidence[
            "metadata_resolution"
        ]
    )

    assert resolution["company_resolved"] is False

    # The materializer must not replace the missing value with
    # the evidence value.
    assert episode.company == "Company Not Specified"


def test_candidate_confidence_is_preserved_as_provenance():
    candidate = make_candidate()

    episode = ExperienceEpisodeMaterializer().materialize(
        candidate
    )

    assert (
        episode.evidence[
            "candidate_confidence"
        ]
        == 0.92
    )


def test_final_confidence_is_calculated_from_final_episode():
    candidate = make_candidate()

    episode = ExperienceEpisodeMaterializer().materialize(
        candidate
    )

    assert 0.0 <= episode.confidence <= 0.99

    # Complete episode should receive the high-confidence
    # score defined by the existing production scorer.
    assert episode.confidence == 0.99


def test_description_and_bullets_are_preserved():
    episode = ExperienceEpisodeMaterializer().materialize(
        make_candidate()
    )

    assert (
        episode.description
        == "Built production machine learning systems."
    )

    assert episode.bullet_points == [
        "Built inference pipelines.",
        "Improved model serving reliability.",
    ]

    assert episode.technologies_mentioned == [
        "Python",
        "PyTorch",
    ]


def test_materialization_is_deterministic():
    candidate = make_candidate()

    materializer = ExperienceEpisodeMaterializer()

    first = materializer.materialize(
        candidate
    )

    second = materializer.materialize(
        candidate
    )

    assert first.model_dump() == second.model_dump()