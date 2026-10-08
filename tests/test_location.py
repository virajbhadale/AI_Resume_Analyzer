from src.extraction.location import LocationExtractor


def test_city_state_location():
    extractor = LocationExtractor()

    result = extractor.detect("Pune, Maharashtra")

    assert result is not None
    location, confidence = result

    assert location == "Pune, Maharashtra"
    assert confidence >= 0.80


def test_city_country_location():
    extractor = LocationExtractor()

    result = extractor.detect("Mumbai, India")

    assert result is not None
    location, confidence = result

    assert location == "Mumbai, India"
    assert confidence >= 0.80


def test_us_location():
    extractor = LocationExtractor()

    result = extractor.detect("San Francisco, CA")

    assert result is not None
    location, confidence = result

    assert location == "San Francisco, CA"
    assert confidence >= 0.80


def test_remote_location():
    extractor = LocationExtractor()

    result = extractor.detect("Remote")

    assert result is not None
    location, confidence = result

    assert location == "Remote"
    assert confidence >= 0.80


def test_hybrid_location():
    extractor = LocationExtractor()

    result = extractor.detect("Hybrid")

    assert result is not None
    location, confidence = result

    assert location == "Hybrid"
    assert confidence >= 0.80


def test_remote_country():
    extractor = LocationExtractor()

    result = extractor.detect("Remote - India")

    assert result is not None
    location, confidence = result

    assert location == "Remote - India"
    assert confidence >= 0.80


def test_skill_list_is_not_location():
    extractor = LocationExtractor()

    assert extractor.detect(
        "Python, SQL, Pandas"
    ) is None


def test_action_sentence_is_not_location():
    extractor = LocationExtractor()

    assert extractor.detect(
        "Built APIs, models, and pipelines"
    ) is None


def test_technical_terms_are_not_location():
    extractor = LocationExtractor()

    assert extractor.detect(
        "Machine Learning, Deep Learning"
    ) is None