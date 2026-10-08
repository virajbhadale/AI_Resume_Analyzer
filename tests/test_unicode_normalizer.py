from src.preprocessing.unicode_normalizer import (
    normalize_unicode,
    normalize_lines,
    normalize_for_matching,
)


def test_normalize_none():
    assert normalize_unicode(None) == ""


def test_normalize_empty():
    assert normalize_unicode("") == ""


def test_normalize_whitespace():
    result = normalize_unicode(
        "  Data   Science\tEngineer  "
    )

    assert result == "Data Science Engineer"


def test_normalize_non_breaking_space():
    result = normalize_unicode(
        "Data\u00A0Scientist"
    )

    assert result == "Data Scientist"


def test_normalize_bullet():
    result = normalize_unicode(
        "â€¢ Built ML pipeline"
    )

    assert "Built ML pipeline" in result


def test_normalize_lines():
    result = normalize_lines(
        "Python\n\n\nMachine Learning\n"
    )

    assert result == [
        "Python",
        "Machine Learning",
    ]


def test_matching_normalization():
    result = normalize_for_matching(
        "  Data   Science Engineer "
    )

    assert result == "data science engineer"


def test_technical_tokens_preserved():
    text = "C++ C# .NET Node.js PyTorch CI/CD"

    result = normalize_unicode(text)

    assert "C++" in result
    assert "C#" in result
    assert ".NET" in result
    assert "Node.js" in result
    assert "PyTorch" in result
    assert "CI/CD" in result