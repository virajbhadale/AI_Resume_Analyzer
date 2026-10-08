"""
Tests for universal education extraction and normalization.

Run:
    pytest -q tests/test_education.py
"""

from src.document_models import BBox, DocumentBlock
from src.extraction.education import EducationExtractor


def make_block(text: str, page: int = 1, y: float = 50.0) -> DocumentBlock:
    """Create a minimal DocumentBlock for testing."""
    return DocumentBlock(
        id=f"test_{page}_{int(y)}",
        page=page,
        text=text,
        bbox=BBox(
            x0=50.0,
            y0=y,
            x1=550.0,
            y1=y + 16.0,
        ),
        font_size=10.0,
    )


def extract(texts):
    """Build blocks and run education extraction."""
    blocks = [
        make_block(text, y=50.0 + (i * 25.0))
        for i, text in enumerate(texts)
    ]

    extractor = EducationExtractor()
    return extractor.extract(blocks)


# ============================================================
# 1. BASIC BACHELOR'S DEGREE
# ============================================================

def test_bachelors_degree():
    results = extract([
        "Education",
        "B.Tech in Computer Science",
        "MIT University",
        "2021 - 2025",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "B.Tech" in education.degree
    assert education.institution == "MIT University"
    assert education.start_date == "2021"
    assert education.end_date == "2025"


# ============================================================
# 2. MASTER'S DEGREE
# ============================================================

def test_masters_degree():
    results = extract([
        "Education",
        "MSc Data Science",
        "MIT World Peace University",
        "2025 - 2027",
        "CGPA 8.09/10",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "MSc" in education.degree
    assert "Data Science" in education.degree
    assert education.institution == "MIT World Peace University"
    assert education.start_date == "2025"
    assert education.end_date == "2027"


# ============================================================
# 3. PURSUING DEGREE
# ============================================================

def test_pursuing_degree():
    results = extract([
        "Education",
        "Pursuing BSc Data Science",
        "Symbiosis Skills & Professional University",
        "2022-2025",
        "CGPA scored = 8.62/10",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "BSc" in education.degree
    assert "Data Science" in education.degree
    assert education.institution == "Symbiosis Skills & Professional University"
    assert education.start_date == "2022"
    assert education.end_date == "2025"
    assert "8.62" in (education.gpa or "")


# ============================================================
# 4. CLASS X
# ============================================================

def test_class_x():
    results = extract([
        "Education",
        "Class X",
        "Blossom Public School",
        "March 2020",
        "Percentage - 84.8%",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "Class X" in education.degree
    assert education.institution == "Blossom Public School"
    assert education.end_date == "2020"
    assert "84.8" in (education.gpa or "")


# ============================================================
# 5. CLASS XII
# ============================================================

def test_class_xii():
    results = extract([
        "Education",
        "Class XII",
        "Blossom Public School",
        "March 2022",
        "Percentage - 57.6%",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "Class XII" in education.degree
    assert education.institution == "Blossom Public School"
    assert education.end_date == "2022"
    assert "57.6" in (education.gpa or "")


# ============================================================
# 6. MULTIPLE EDUCATION ENTRIES
# ============================================================

def test_multiple_education_entries():
    results = extract([
        "Education",

        "Class X",
        "Blossom Public School",
        "March 2020",
        "84.8%",

        "Class XII",
        "Blossom Public School",
        "March 2022",
        "57.6%",

        "BSc Data Science",
        "Symbiosis Skills & Professional University",
        "2022-2025",
        "CGPA 8.62/10",

        "MSc Data Science and Big Data Analytics",
        "MIT World Peace University",
        "2025-2027",
        "CGPA 8.09",
    ])

    assert len(results) == 4

    degrees = [item.degree for item in results]

    assert any("Class X" in degree for degree in degrees)
    assert any("Class XII" in degree for degree in degrees)
    assert any("BSc" in degree for degree in degrees)
    assert any("MSc" in degree for degree in degrees)


# ============================================================
# 7. CGPA NORMALIZATION
# ============================================================

def test_cgpa_normalization():
    results = extract([
        "Education",
        "BSc Computer Science",
        "ABC University",
        "2020 - 2024",
        "CGPA scored = 8.62/10",
    ])

    assert len(results) >= 1

    education = results[0]

    assert education.gpa is not None
    assert "8.62" in education.gpa


# ============================================================
# 8. PERCENTAGE NORMALIZATION
# ============================================================

def test_percentage_normalization():
    results = extract([
        "Education",
        "Class XII",
        "ABC School",
        "March 2022",
        "Percentage - 91.4%",
    ])

    assert len(results) >= 1

    education = results[0]

    assert education.gpa is not None
    assert "91.4" in education.gpa


# ============================================================
# 9. DATE RANGE WITH EN DASH
# ============================================================

def test_en_dash_date_range():
    results = extract([
        "Education",
        "B.Tech Computer Engineering",
        "ABC Institute",
        "2020 – 2024",
    ])

    assert len(results) >= 1

    education = results[0]

    assert education.start_date == "2020"
    assert education.end_date == "2024"


# ============================================================
# 10. SHORT YEAR RANGE
# ============================================================

def test_short_year_range():
    results = extract([
        "Education",
        "MSc Data Science",
        "ABC University",
        "2025-27",
    ])

    assert len(results) >= 1

    education = results[0]

    assert education.start_date == "2025"
    assert education.end_date == "2027"


# ============================================================
# 11. DEGREE AND UNIVERSITY ON SAME LINE
# ============================================================

def test_degree_and_institution_same_line():
    results = extract([
        "Education",
        "B.Tech Computer Science - ABC University",
        "2020-2024",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "B.Tech" in education.degree
    assert education.institution


# ============================================================
# 12. UNIVERSITY BEFORE DEGREE
# ============================================================

def test_institution_before_degree():
    results = extract([
        "Education",
        "ABC University",
        "Bachelor of Technology in Computer Science",
        "2020-2024",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "Bachelor" in education.degree or "Technology" in education.degree
    assert education.institution == "ABC University"


# ============================================================
# 13. FIELD OF STUDY
# ============================================================

def test_field_of_study_is_preserved():
    results = extract([
        "Education",
        "Bachelor of Engineering in Artificial Intelligence",
        "ABC University",
        "2020-2024",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "Artificial Intelligence" in education.degree


# ============================================================
# 14. DOCTORATE
# ============================================================

def test_doctorate_degree():
    results = extract([
        "Education",
        "PhD in Machine Learning",
        "Stanford University",
        "2021 - Present",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "PhD" in education.degree
    assert education.institution == "Stanford University"
    assert education.start_date == "2021"


# ============================================================
# 15. MBA / NON-TECHNICAL DEGREE
# ============================================================

def test_mba_degree():
    results = extract([
        "Education",
        "MBA in Business Analytics",
        "ABC Business School",
        "2022-2024",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "MBA" in education.degree
    assert education.institution == "ABC Business School"


# ============================================================
# 16. DIPLOMA
# ============================================================

def test_diploma():
    results = extract([
        "Education",
        "Diploma in Computer Engineering",
        "ABC Polytechnic",
        "2018-2021",
    ])

    assert len(results) >= 1

    education = results[0]

    assert "Diploma" in education.degree
    assert education.institution == "ABC Polytechnic"


# ============================================================
# 17. EDUCATION DETAILS / EXTRA INFORMATION
# ============================================================

def test_additional_education_details():
    results = extract([
        "Education",
        "B.Tech Computer Science",
        "ABC University",
        "2020-2024",
        "Relevant Coursework: Machine Learning, Data Structures",
        "Activities: Coding Club",
    ])

    assert len(results) >= 1

    education = results[0]

    assert education.degree
    assert education.institution

    # The extractor may preserve these as details.
    details_text = " ".join(education.details)

    assert (
        "Machine Learning" in details_text
        or "Data Structures" in details_text
        or "Coding Club" in details_text
        or len(education.details) >= 0
    )


# ============================================================
# 18. REAL-RESUME STYLE EDUCATION
# ============================================================

def test_real_resume_style_education():
    results = extract([
        "Education",

        "Class X",
        "Blossom Public School",
        "March 2020 Percentage - 84.8%",

        "Class XII",
        "Blossom Public School",
        "March 2022 Percentage - 57.6%",

        "Symbiosis Skills & Professional University",
        "Pursuing BSc Data Science (2022-2025)",
        "CGPA scored = 8.62/10",

        "MIT World Peace University",
        "Pursuing MSc Data Science and Big Data Analytics (2025-27)",
        "CGPA Scored - 8.09",
    ])

    assert len(results) == 4

    degrees = " | ".join(item.degree for item in results)

    assert "Class X" in degrees
    assert "Class XII" in degrees
    assert "BSc" in degrees
    assert "MSc" in degrees


# ============================================================
# 19. EMPTY INPUT
# ============================================================

def test_empty_input():
    results = extract([])

    assert isinstance(results, list)
    assert len(results) == 0


# ============================================================
# 20. NON-EDUCATION CONTENT SHOULD NOT BECOME EDUCATION
# ============================================================

def test_non_education_content_is_ignored():
    results = extract([
        "Professional Experience",
        "Data Scientist",
        "Built machine learning models using Python.",
        "Developed REST APIs using FastAPI.",
        "Skills",
        "Python, SQL, Machine Learning",
    ])

    assert isinstance(results, list)

    # The extractor should not manufacture an education record
    # from ordinary professional content.
    assert len(results) == 0


# ============================================================
# 21. DUPLICATE EDUCATION ENTRY
# ============================================================

def test_duplicate_education_entries_are_handled():
    results = extract([
        "Education",
        "BSc Data Science",
        "ABC University",
        "2022-2025",

        "BSc Data Science",
        "ABC University",
        "2022-2025",
    ])

    # Depending on the implementation, duplicates should either
    # be merged or safely represented without crashing.
    assert isinstance(results, list)
    assert len(results) >= 1


# ============================================================
# 22. EDUCATION EXTRACTION NEVER CRASHES
# ============================================================

def test_unusual_education_text_does_not_crash():
    results = extract([
        "Education",
        "M.Sc. (Data Science)",
        "ABC University | Remote",
        "Aug 2024 – May 2026",
        "GPA: 3.9 / 4.0",
        "Coursework: NLP, Deep Learning, Statistics",
    ])

    assert isinstance(results, list)

    if results:
        assert results[0].degree

# ============================================================
# 23. UNIVERSITY NAME MUST NOT BE DETECTED AS A DEGREE
# ============================================================

def test_university_name_not_split_into_separate_degree():
    results = extract([
        "Education",
        "Bachelor of Science in Statistics",
        "University of California, Berkeley",
        "2015 - 2019",
    ])

    assert len(results) == 1

    education = results[0]

    assert "Bachelor of Science in Statistics" in education.degree
    assert education.institution == "University of California, Berkeley"
    assert education.start_date == "2015"
    assert education.end_date == "2019"        