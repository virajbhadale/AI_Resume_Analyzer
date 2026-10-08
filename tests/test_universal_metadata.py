"""
Universal metadata-resolution contract tests.

Phase:
    18C.6 — Metadata Resolution

Purpose:
    Verify that MetadataResolver preserves, normalizes and safely resolves
    metadata already present on an ExperienceEpisode.

Important:
    MetadataResolver operates on an ExperienceEpisode. Therefore document
    ordering, candidate competition and block-level association are NOT
    tested here. Those responsibilities belong to candidate generation /
    evidence fusion.

Contract:
    - Never invent company/location/date values.
    - Preserve valid metadata.
    - Normalize safe formatting noise.
    - Separate embedded explicit location from company where supported.
    - Preserve compatibility sentinels used by the current ExperienceEpisode
      model contract.
"""

from src.inference.metadata import MetadataResolver
from src.document_models import ExperienceEpisode


ROLE_MISSING = "Role Not Specified"
COMPANY_MISSING = "Company Not Specified"


def make_episode(
    *,
    role=ROLE_MISSING,
    company=COMPANY_MISSING,
    location=None,
    start_date=None,
    end_date=None,
    is_current=False,
    description="",
    bullet_points=None,
):
    """
    Build an ExperienceEpisode using the current public model contract.

    The current ExperienceEpisode model requires role/company strings, so
    missing values are represented using the existing compatibility
    sentinels.
    """
    return ExperienceEpisode(
        id="metadata-test-episode",
        role=role,
        company=company,
        location=location,
        start_date=start_date,
        end_date=end_date,
        is_current=is_current,
        description=description,
        bullet_points=bullet_points or [],
        technologies_mentioned=[],
        confidence=0.0,
        evidence={},
    )


# ============================================================================
# 1. COMPLETE METADATA
# ============================================================================


def test_complete_metadata_is_preserved():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        location="Pune, India",
        start_date="Jan 2023",
        end_date="Present",
        is_current=True,
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Data Scientist"
    assert result.company == "ABC Technologies"
    assert result.location == "Pune, India"
    assert result.start_date == "Jan 2023"
    assert result.end_date == "Present"
    assert result.is_current is True


# ============================================================================
# 2. MISSING COMPANY
# ============================================================================


def test_missing_company_is_preserved_as_missing():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Data Scientist",
        company=COMPANY_MISSING,
        location="Pune, India",
        start_date="Jan 2023",
        end_date="Present",
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Data Scientist"
    assert result.company == COMPANY_MISSING
    assert result.location == "Pune, India"


# ============================================================================
# 3. MISSING LOCATION
# ============================================================================


def test_missing_location_is_not_invented():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        location=None,
        start_date="Jan 2023",
        end_date="Present",
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Data Scientist"
    assert result.company == "ABC Technologies"
    assert result.location is None


# ============================================================================
# 4. MISSING DATES
# ============================================================================


def test_missing_dates_are_not_invented():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Machine Learning Engineer",
        company="ABC Technologies",
        location="Pune, India",
        start_date=None,
        end_date=None,
        is_current=False,
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Machine Learning Engineer"
    assert result.company == "ABC Technologies"
    assert result.location == "Pune, India"
    assert result.start_date is None
    assert result.end_date is None


# ============================================================================
# 5. ROLE + LOCATION + DATE WITHOUT COMPANY
# ============================================================================


def test_role_location_date_does_not_create_company():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Machine Learning Engineer",
        company=COMPANY_MISSING,
        location="Pune, India",
        start_date="Jun 2023",
        end_date="Aug 2024",
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Machine Learning Engineer"
    assert result.location == "Pune, India"
    assert result.company == COMPANY_MISSING


# ============================================================================
# 6. COMPANY + LOCATION
# ============================================================================


def test_company_and_location_are_preserved():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Data Analyst",
        company="ABC Analytics",
        location="Mumbai, India",
    )

    result = resolver.resolve_episode(episode)

    assert result.company == "ABC Analytics"
    assert result.location == "Mumbai, India"


# ============================================================================
# 7. EMBEDDED EXPLICIT LOCATION
# ============================================================================


def test_embedded_location_is_separated_from_company():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Data Analyst Intern",
        company="ByzWiz Infratech Pvt Ltd. (Location - Pune, India)",
        location=None,
        start_date="Jan 2023",
        end_date="May 2023",
    )

    result = resolver.resolve_episode(episode)

    assert result.company == "ByzWiz Infratech Pvt Ltd."
    assert result.location == "Pune, India"


# ============================================================================
# 8. EXPLICIT LOCATION PREFIX
# ============================================================================


def test_explicit_location_prefix_is_normalized():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Data Analyst",
        company="ABC Analytics",
        location="Location - Pune, India",
    )

    result = resolver.resolve_episode(episode)

    assert result.company == "ABC Analytics"
    assert result.location == "Pune, India"


# ============================================================================
# 9. LOCATION DOES NOT BECOME COMPANY
# ============================================================================


def test_location_value_is_not_reclassified_as_company():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Data Scientist",
        company=COMPANY_MISSING,
        location="Thailand",
        start_date="Jan 2022",
        end_date="Dec 2022",
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Data Scientist"
    assert result.location == "Thailand"
    assert result.company == COMPANY_MISSING


# ============================================================================
# 10. COMPANY DOES NOT DISAPPEAR WHEN LOCATION IS MISSING
# ============================================================================


def test_company_survives_missing_location():
    resolver = MetadataResolver()

    episode = make_episode(
        role="AI Engineer",
        company="CognitiveCore AI",
        location=None,
        start_date="Feb 2024",
        end_date="Present",
        is_current=True,
    )

    result = resolver.resolve_episode(episode)

    assert result.company == "CognitiveCore AI"
    assert result.location is None
    assert result.is_current is True


# ============================================================================
# 11. SAME COMPANY WITH DIFFERENT ROLES
# ============================================================================


def test_metadata_resolution_does_not_modify_role_identity():
    resolver = MetadataResolver()

    first = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        location="Pune, India",
        start_date="Jan 2022",
        end_date="Dec 2022",
    )

    second = make_episode(
        role="Senior Data Scientist",
        company="ABC Technologies",
        location="Pune, India",
        start_date="Jan 2023",
        end_date="Present",
        is_current=True,
    )

    first_result = resolver.resolve_episode(first)
    second_result = resolver.resolve_episode(second)

    assert first_result.role == "Data Scientist"
    assert second_result.role == "Senior Data Scientist"

    assert first_result.company == "ABC Technologies"
    assert second_result.company == "ABC Technologies"

    assert first_result.start_date == "Jan 2022"
    assert second_result.start_date == "Jan 2023"


# ============================================================================
# 12. NO METADATA
# ============================================================================


def test_completely_missing_metadata_is_not_invented():
    resolver = MetadataResolver()

    episode = make_episode(
        role=ROLE_MISSING,
        company=COMPANY_MISSING,
        location=None,
        start_date=None,
        end_date=None,
        is_current=False,
    )

    result = resolver.resolve_episode(episode)

    assert result.role == ROLE_MISSING
    assert result.company == COMPANY_MISSING
    assert result.location is None
    assert result.start_date is None
    assert result.end_date is None


# ============================================================================
# 13. WHITESPACE NORMALIZATION
# ============================================================================


def test_metadata_whitespace_is_normalized():
    resolver = MetadataResolver()

    episode = make_episode(
        role="  Data Scientist  ",
        company="  ABC Technologies  ",
        location="  Pune, India  ",
        start_date="  Jan 2023  ",
        end_date="  Present  ",
    )

    result = resolver.resolve_episode(episode)

    assert result.role == "Data Scientist"
    assert result.company == "ABC Technologies"
    assert result.location == "Pune, India"
    assert result.start_date == "Jan 2023"
    assert result.end_date == "Present"


# ============================================================================
# 14. CURRENT FLAG IS PRESERVED
# ============================================================================


def test_current_status_is_preserved():
    resolver = MetadataResolver()

    episode = make_episode(
        role="AI Engineer",
        company="ABC Technologies",
        start_date="Jan 2024",
        end_date="Present",
        is_current=True,
    )

    result = resolver.resolve_episode(episode)

    assert result.is_current is True
    assert result.end_date == "Present"


# ============================================================================
# 15. DESCRIPTION AND BULLETS ARE NOT DESTROYED
# ============================================================================


def test_metadata_resolution_preserves_description_content():
    resolver = MetadataResolver()

    bullets = [
        "Built machine learning models.",
        "Developed automated data pipelines.",
    ]

    episode = make_episode(
        role="Machine Learning Engineer",
        company="ABC Technologies",
        location="Pune, India",
        start_date="Jan 2023",
        end_date="Present",
        description="\n".join(bullets),
        bullet_points=bullets,
    )

    result = resolver.resolve_episode(episode)

    assert result.description == "\n".join(bullets)
    assert result.bullet_points == bullets


# ============================================================================
# 16. RESOLUTION IS IDEMPOTENT
# ============================================================================


def test_metadata_resolution_is_idempotent():
    resolver = MetadataResolver()

    episode = make_episode(
        role="Data Scientist",
        company="ABC Technologies",
        location="Pune, India",
        start_date="Jan 2023",
        end_date="Present",
        is_current=True,
    )

    first = resolver.resolve_episode(episode)
    second = resolver.resolve_episode(first)

    assert second.role == first.role
    assert second.company == first.company
    assert second.location == first.location
    assert second.start_date == first.start_date
    assert second.end_date == first.end_date
    assert second.is_current == first.is_current