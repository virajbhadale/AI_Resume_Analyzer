from src.document_models import ExperienceEpisode
from src.inference.metadata import MetadataResolver


def test_metadata_preserves_valid_episode():
    resolver = MetadataResolver()

    episode = ExperienceEpisode(
        id="exp_001",
        role="AI Engineer Intern",
        company="Marva AI",
        location="Pune, Maharashtra",
        start_date="06 Jan 2025",
        end_date="17 May 2025",
        is_current=False,
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "AI Engineer Intern"
    assert result.company == "Marva AI"
    assert result.location == "Pune, Maharashtra"
    assert result.start_date == "06 Jan 2025"
    assert result.end_date == "17 May 2025"
    assert result.is_current is False


def test_metadata_normalizes_current_status():
    resolver = MetadataResolver()

    episode = ExperienceEpisode(
        id="exp_002",
        role="Data Scientist",
        company="Nexus Analytics",
        end_date="Current",
        is_current=False,
    )

    result = resolver.resolve_episode(episode)

    assert result.end_date == "Present"
    assert result.is_current is True


def test_metadata_normalizes_present_status():
    resolver = MetadataResolver()

    episode = ExperienceEpisode(
        id="exp_003",
        role="ML Engineer",
        company="Apex AI",
        end_date="Present",
        is_current=True,
    )

    result = resolver.resolve_episode(episode)

    assert result.end_date == "Present"
    assert result.is_current is True


def test_metadata_normalizes_location():
    resolver = MetadataResolver()

    episode = ExperienceEpisode(
        id="exp_004",
        role="Developer",
        company="TechCorp",
        location="  Pune ,   Maharashtra  ",
    )

    result = resolver.resolve_episode(episode)

    assert result.location == "Pune, Maharashtra"


def test_metadata_handles_missing_values():
    resolver = MetadataResolver()

    episode = ExperienceEpisode(
        id="exp_005",
        role="",
        company="None",
        location="N/A",
        start_date="",
        end_date="",
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Role Not Specified"
    assert result.company == "Company Not Specified"
    assert result.location is None
    assert result.start_date is None
    assert result.end_date is None
    assert result.is_current is False


def test_metadata_resolves_multiple_episodes():
    resolver = MetadataResolver()

    episodes = [
        ExperienceEpisode(
            id="exp_001",
            role="Data Analyst Intern",
            company="ComplyHappy",
            location="Mumbai, India",
        ),
        ExperienceEpisode(
            id="exp_002",
            role="AI Engineer Intern",
            company="Marva AI",
            location="Pune, Maharashtra",
        ),
    ]

    results = resolver.resolve(episodes)

    assert len(results) == 2
    assert results[0].company == "ComplyHappy"
    assert results[1].company == "Marva AI"


def test_metadata_records_resolution_evidence():
    resolver = MetadataResolver()

    episode = ExperienceEpisode(
        id="exp_006",
        role="AI Engineer",
        company="Marva AI",
        location="Pune, Maharashtra",
        start_date="2025",
        end_date="Present",
        is_current=True,
    )

    result = resolver.resolve_episode(episode)

    evidence = result.evidence["metadata_resolution"]

    assert evidence["role_resolved"] is True
    assert evidence["company_resolved"] is True
    assert evidence["location_resolved"] is True
    assert evidence["start_date_resolved"] is True
    assert evidence["end_date_resolved"] is True
    assert evidence["current_status_resolved"] is True

# ============================================================================
# PHASE 18C.6 — UNIVERSAL METADATA CONTRACT
# ============================================================================

from src.document_models import ExperienceEpisode
from src.inference.metadata import MetadataResolver


def _phase18c6_episode(
    *,
    role="Role Not Specified",
    company="Company Not Specified",
    location=None,
    start_date=None,
    end_date=None,
    is_current=False,
):
    return ExperienceEpisode(
        id="phase18c6",
        role=role,
        company=company,
        location=location,
        start_date=start_date,
        end_date=end_date,
        is_current=is_current,
        description="",
        bullet_points=[],
        technologies_mentioned=[],
        confidence=0.0,
        evidence={},
    )


def test_phase18c6_does_not_invent_company_from_location():
    resolver = MetadataResolver()

    episode = _phase18c6_episode(
        role="Data Scientist",
        company="Company Not Specified",
        location="Thailand",
        start_date="Jan 2022",
        end_date="Dec 2022",
    )

    result = resolver.resolve_episode(episode)

    assert result.location == "Thailand"
    assert result.company == "Company Not Specified"


def test_phase18c6_preserves_missing_location():
    resolver = MetadataResolver()

    episode = _phase18c6_episode(
        role="Data Scientist",
        company="ABC Technologies",
        location=None,
    )

    result = resolver.resolve_episode(episode)

    assert result.company == "ABC Technologies"
    assert result.location is None


def test_phase18c6_preserves_missing_dates():
    resolver = MetadataResolver()

    episode = _phase18c6_episode(
        role="Data Scientist",
        company="ABC Technologies",
    )

    result = resolver.resolve_episode(episode)

    assert result.start_date is None
    assert result.end_date is None


def test_phase18c6_embedded_location_is_split():
    resolver = MetadataResolver()

    episode = _phase18c6_episode(
        role="Data Analyst Intern",
        company=(
            "ByzWiz Infratech Pvt Ltd. "
            "(Location - Pune, India)"
        ),
    )

    result = resolver.resolve_episode(episode)

    assert result.company == "ByzWiz Infratech Pvt Ltd."
    assert result.location == "Pune, India"


def test_phase18c6_resolution_does_not_mutate_role():
    resolver = MetadataResolver()

    episode = _phase18c6_episode(
        role="Senior Machine Learning Engineer",
        company="ABC Technologies",
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Senior Machine Learning Engineer"    