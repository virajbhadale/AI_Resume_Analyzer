"""
End-to-end validation of experience extraction.

This test intentionally prints the final experience objects produced by
the real resume analysis pipeline.

Purpose:
    Verify that experience segmentation, metadata resolution,
    technologies, confidence, and final resume integration all survive
    the complete pipeline.

This is a validation/debug test, not a permanent regression test yet.
"""

from pprint import pprint

from src.pipeline import analyze_resume


# ============================================================
# CONFIGURATION
# ============================================================

RESUME_PATH = "data/samples/Viraj_Resume.pdf"


# ============================================================
# HELPER
# ============================================================

def _get_value(obj, key, default=None):
    """Read a value from either a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _print_experience(index, experience):
    """Print one final extracted experience."""

    print("\n" + "=" * 70)
    print(f"EXPERIENCE {index}")
    print("=" * 70)

    fields = [
        "role",
        "company",
        "location",
        "start_date",
        "end_date",
        "is_current",
        "description",
        "bullet_points",
        "technologies_mentioned",
        "confidence",
        "evidence",
    ]

    for field in fields:
        value = _get_value(experience, field)

        print(f"\n{field}:")
        pprint(value, sort_dicts=False)


# ============================================================
# END-TO-END TEST
# ============================================================

def test_real_resume_experience_extraction():

    print("\n" + "#" * 70)
    print("# END-TO-END EXPERIENCE VALIDATION")
    print("#" * 70)

    print(f"\nResume: {RESUME_PATH}")

    # --------------------------------------------------------
    # Run the REAL production pipeline
    # --------------------------------------------------------

    result = analyze_resume(RESUME_PATH)

    # --------------------------------------------------------
    # Find final structured resume
    # --------------------------------------------------------

    if isinstance(result, dict):

        print("\nResult keys:")
        print(list(result.keys()))

        resume = (
            result.get("resume")
            or result.get("structured_resume")
            or result
        )

    else:
        resume = result

    # --------------------------------------------------------
    # Get experiences
    # --------------------------------------------------------

    experiences = _get_value(
        resume,
        "experiences",
        [],
    )

    print("\n" + "-" * 70)
    print(f"TOTAL EXPERIENCES: {len(experiences)}")
    print("-" * 70)

    # --------------------------------------------------------
    # Print every experience
    # --------------------------------------------------------

    for index, experience in enumerate(
        experiences,
        start=1,
    ):
        _print_experience(
            index,
            experience,
        )

    # --------------------------------------------------------
    # Basic sanity assertions
    # --------------------------------------------------------

    assert experiences is not None

    print("\n" + "#" * 70)
    print("# END-TO-END VALIDATION COMPLETED")
    print("#" * 70)