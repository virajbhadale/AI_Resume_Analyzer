from src.inference.anchors import AnchorDetector


HEADERS = [
    "Senior Data Scientist & Machine Learning Software Engineer (Generative AI) PURELOGICS",
    "Senior Data Scientist Machine Learning & Gen AI Engineer (Volunteer) NVIDIA",
    "Senior Data Scientist & Machine Learning Engineer BLOCBELT",
    "AI Data Scientist Kaggle Master Mentor (Part-Time, 20 hours/week)",
    "Data Scientist",
    "Thailand",
    "Crypto-Express",
    "Artificial Intelligence Engineer",
    "Lahore, Pakistan",
    "Pakistan Freelancing Training Center",
]


def test_jillani_header_anchor_forensics():

    detector = AnchorDetector()

    print("\n" + "=" * 100)
    print("JILLANI HEADER ANCHOR FORENSICS")
    print("=" * 100)

    for text in HEADERS:

        result = detector.analyze_block(
            text,
            is_bold=False,
        )

        print("\n" + "-" * 100)
        print(f"TEXT: {text!r}")
        print("-" * 100)

        print("role:", getattr(result, "role", None))
        print(
            "role_confidence:",
            getattr(result, "role_confidence", None),
        )

        print("company:", getattr(result, "company", None))
        print(
            "company_confidence:",
            getattr(result, "company_confidence", None),
        )

        print("location:", getattr(result, "location", None))
        print(
            "location_confidence:",
            getattr(result, "location_confidence", "N/A"),
        )

        print("start_date:", getattr(result, "start_date", None))
        print("end_date:", getattr(result, "end_date", None))
        print("is_current:", getattr(result, "is_current", None))
        print("is_anchor_head:", getattr(result, "is_anchor_head", None))
        print("confidence:", getattr(result, "confidence", None))

        print("available fields:")
        print(
            list(
                getattr(result, "model_fields", {}).keys()
            )
        )

        print("signals:")

        for signal in getattr(result, "supporting_evidence", []):
            print(
                "  SUPPORT:",
                getattr(signal, "signal_type", None),
                getattr(signal, "value", None),
                getattr(signal, "confidence", None),
            )

        for signal in getattr(result, "contradictory_evidence", []):
            print(
                "  CONTRADICT:",
                getattr(signal, "signal_type", None),
                getattr(signal, "value", None),
                getattr(signal, "confidence", None),
            )

    assert True