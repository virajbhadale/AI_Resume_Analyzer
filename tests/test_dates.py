from src.extraction.dates import DateExtractor


def test_month_year_to_present():
    extractor = DateExtractor()

    result = extractor.extract_date_range(
        "Jan 2023 - Present"
    )

    assert result is not None
    assert result["start_date"] == "Jan 2023"
    assert result["end_date"] == "Present"
    assert result["is_current"] is True


def test_full_date_range():
    extractor = DateExtractor()

    result = extractor.extract_date_range(
        "19 Jun 2023 - 11 Aug 2023"
    )

    assert result is not None
    assert result["start_date"] == "19 Jun 2023"
    assert result["end_date"] == "11 Aug 2023"
    assert result["is_current"] is False


def test_year_only_range():
    extractor = DateExtractor()

    result = extractor.extract_date_range(
        "2021 - 2023"
    )

    assert result is not None
    assert result["start_date"] == "2021"
    assert result["end_date"] == "2023"


def test_month_year_range():
    extractor = DateExtractor()

    result = extractor.extract_date_range(
        "Jun 2021 - Dec 2022"
    )

    assert result is not None
    assert result["start_date"] == "Jun 2021"
    assert result["end_date"] == "Dec 2022"


def test_since_date():
    extractor = DateExtractor()

    result = extractor.extract_date_range(
        "Since Jan 2024"
    )

    assert result is not None
    assert result["start_date"] == "Jan 2024"
    assert result["end_date"] == "Present"
    assert result["is_current"] is True


def test_from_date():
    extractor = DateExtractor()

    result = extractor.extract_date_range(
        "From 2022"
    )

    assert result is not None
    assert result["start_date"] == "2022"
    assert result["end_date"] == "Present"
    assert result["is_current"] is True


def test_current_terms():
    extractor = DateExtractor()

    for term in [
        "Present",
        "Current",
        "Ongoing",
        "Now",
        "Today",
    ]:
        result = extractor.extract_date_range(
            f"Jan 2024 - {term}"
        )

        assert result is not None
        assert result["is_current"] is True


def test_no_date():
    extractor = DateExtractor()

    result = extractor.extract_date_range(
        "Built machine learning models using Python."
    )

    assert result is None


def test_has_date_signal():
    extractor = DateExtractor()

    assert extractor.has_date_signal(
        "Machine Learning Engineer 2021 - 2023"
    )

    assert extractor.has_date_signal(
        "Worked from Jan 2024"
    )

    assert not extractor.has_date_signal(
        "Python, SQL, Pandas"
    )