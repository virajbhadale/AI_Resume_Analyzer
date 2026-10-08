from src.analysis.skill_gap import calculate_skill_gap


def test_skill_gap_basic():
    result = calculate_skill_gap(
        matched_core=["Python", "FastAPI"],
        matched_secondary=["PyTorch"],
        missing_core=["Docker"],
        missing_secondary=["Kubernetes"],
    )

    assert result.matched_core == ["Python", "FastAPI"]
    assert result.matched_secondary == ["PyTorch"]

    assert result.missing_core == ["Docker"]
    assert result.missing_secondary == ["Kubernetes"]

    assert result.total_matched == 3
    assert result.total_missing == 2

    assert result.coverage_percentage == 60.0


def test_skill_gap_calculates_core_and_secondary_coverage():
    result = calculate_skill_gap(
        matched_core=["Python", "FastAPI"],
        matched_secondary=["PyTorch"],
        missing_core=["Docker"],
        missing_secondary=["Kubernetes"],
    )

    assert result.core_coverage_percentage == 66.67
    assert result.secondary_coverage_percentage == 50.0


def test_skill_gap_builds_aggregated_skill_lists():
    result = calculate_skill_gap(
        matched_core=["Python", "FastAPI"],
        matched_secondary=["PyTorch"],
        missing_core=["Docker"],
        missing_secondary=["Kubernetes"],
    )

    assert result.matched_skills == [
        "Python",
        "FastAPI",
        "PyTorch",
    ]

    assert result.missing_skills == [
        "Docker",
        "Kubernetes",
    ]

    assert result.required_skills == [
        "Python",
        "FastAPI",
        "PyTorch",
        "Docker",
        "Kubernetes",
    ]


def test_skill_gap_removes_duplicates_case_insensitively():
    result = calculate_skill_gap(
        matched_core=["Python", "python", " FastAPI "],
        matched_secondary=["PYTHON", "PyTorch"],
        missing_core=["Docker"],
        missing_secondary=["docker", "Kubernetes"],
    )

    assert result.matched_core == [
        "Python",
        "FastAPI",
    ]

    assert result.matched_secondary == [
        "PYTHON",
        "PyTorch",
    ]

    assert result.missing_core == [
        "Docker",
    ]

    assert result.missing_secondary == [
        "docker",
        "Kubernetes",
    ]

    assert result.matched_skills == [
        "Python",
        "FastAPI",
        "PyTorch",
    ]

    assert result.missing_skills == [
        "Docker",
        "Kubernetes",
    ]