"""
Tests for Resume Profile extraction.

Run:
    pytest -q tests/test_resume_profile.py
"""

from src.extraction.resume_profile import ResumeProfileExtractor


def extract(
    summary="",
    experience_roles=None,
    skills=None,
    full_text="",
):
    extractor = ResumeProfileExtractor()

    return extractor.extract(
        summary=summary,
        experience_roles=experience_roles or [],
        skills=skills or [],
        full_text=full_text,
    )


# ============================================================
# 1. TITLE FROM EXPERIENCE ROLE
# ============================================================

def test_title_from_experience_role():
    title, domain = extract(
        experience_roles=["Machine Learning Engineer"]
    )

    assert title == "Machine Learning Engineer"


# ============================================================
# 2. EXPERIENCE ROLE HAS PRIORITY
# ============================================================

def test_experience_role_has_priority_over_summary():
    title, domain = extract(
        summary="Passionate Data Scientist with experience in AI.",
        experience_roles=["Machine Learning Engineer"],
    )

    assert title == "Machine Learning Engineer"


# ============================================================
# 3. TITLE FROM SUMMARY
# ============================================================

def test_title_from_summary():
    title, domain = extract(
        summary="I am a Data Scientist specializing in predictive modeling."
    )

    assert title == "Data Scientist"


# ============================================================
# 4. TITLE FROM FULL RESUME TEXT
# ============================================================

def test_title_from_full_text():
    title, domain = extract(
        full_text="Professional Experience\nSoftware Engineer\nBuilt backend systems."
    )

    assert title == "Software Engineer"


# ============================================================
# 5. NO TITLE EVIDENCE
# ============================================================

def test_no_title_returns_professional():
    title, domain = extract(
        summary="Experienced professional with strong communication skills."
    )

    assert title == "Professional"


# ============================================================
# 6. DATA SCIENCE DOMAIN
# ============================================================

def test_data_science_domain():
    title, domain = extract(
        experience_roles=["Data Scientist"],
        skills=["Python", "Pandas", "Scikit-Learn"],
    )

    assert domain == "Data Science"


# ============================================================
# 7. ARTIFICIAL INTELLIGENCE DOMAIN
# ============================================================

def test_artificial_intelligence_domain():
    title, domain = extract(
        summary="AI Engineer working on computer vision and generative AI."
    )

    assert domain == "Artificial Intelligence"


# ============================================================
# 8. SOFTWARE ENGINEERING DOMAIN
# ============================================================

def test_software_engineering_domain():
    title, domain = extract(
        experience_roles=["Software Engineer"],
        skills=["JavaScript", "FastAPI"],
    )

    assert domain == "Software Engineering"


# ============================================================
# 9. DATA ENGINEERING DOMAIN
# ============================================================

def test_data_engineering_domain():
    title, domain = extract(
        experience_roles=["Data Engineer"],
        skills=["Apache Spark", "ETL", "Data Pipelines"],
    )

    assert domain == "Data Engineering"


# ============================================================
# 10. DATA ANALYTICS DOMAIN
# ============================================================

def test_data_analytics_domain():
    title, domain = extract(
        experience_roles=["Data Analyst"],
        skills=["Tableau", "Power BI"],
    )

    assert domain == "Data Analytics"


# ============================================================
# 11. MARKETING DOMAIN
# ============================================================

def test_marketing_domain():
    title, domain = extract(
        summary="Digital marketing specialist with SEO experience."
    )

    assert domain == "Marketing"


# ============================================================
# 12. FINANCE DOMAIN
# ============================================================

def test_finance_domain():
    title, domain = extract(
        experience_roles=["Financial Analyst"],
        skills=["Accounting"],
    )

    assert domain == "Finance"


# ============================================================
# 13. HUMAN RESOURCES DOMAIN
# ============================================================

def test_human_resources_domain():
    title, domain = extract(
        summary="HR manager specializing in recruitment and talent acquisition."
    )

    assert domain == "Human Resources"


# ============================================================
# 14. SALES DOMAIN
# ============================================================

def test_sales_domain():
    title, domain = extract(
        experience_roles=["Sales Manager"],
        full_text="Business development and client acquisition."
    )

    assert domain == "Sales"


# ============================================================
# 15. OPERATIONS DOMAIN
# ============================================================

def test_operations_domain():
    title, domain = extract(
        experience_roles=["Operations Manager"],
        full_text="Responsible for supply chain and logistics."
    )

    assert domain == "Operations"


# ============================================================
# 16. GENERAL DOMAIN
# ============================================================

def test_general_domain_when_no_domain_evidence():
    title, domain = extract(
        summary="Experienced professional with strong communication skills."
    )

    assert domain == "General"


# ============================================================
# 17. EMPTY INPUT
# ============================================================

def test_empty_input():
    title, domain = extract()

    assert title == "Professional"
    assert domain == "General"


# ============================================================
# 18. TITLE WITH SENIORITY
# ============================================================

def test_title_with_seniority():
    title, domain = extract(
        summary="Senior Data Scientist with experience in machine learning."
    )

    assert title == "Senior Data Scientist"


# ============================================================
# 19. AI ENGINEER TITLE
# ============================================================

def test_ai_engineer_title():
    title, domain = extract(
        experience_roles=["AI Engineer"]
    )

    assert title == "AI Engineer"
    assert domain == "Artificial Intelligence"


# ============================================================
# 20. REALISTIC AI/ML RESUME PROFILE
# ============================================================

def test_realistic_ai_ml_resume_profile():
    title, domain = extract(
        summary=(
            "AI Engineer specializing in machine learning, "
            "generative AI and large language models."
        ),
        experience_roles=[
            "AI Engineer",
            "Machine Learning Engineer",
        ],
        skills=[
            "Python",
            "PyTorch",
            "LangChain",
            "RAG",
        ],
        full_text=(
            "Built machine learning systems and "
            "retrieval augmented generation applications."
        ),
    )

    assert title == "AI Engineer"
    assert domain == "Artificial Intelligence"