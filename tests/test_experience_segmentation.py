"""Regression tests for Experience Segmentation to guarantee separate experiences NEVER merge."""
import pytest
from src.document_models import DocumentBlock, BBox
from src.inference.segmenter import ExperienceSegmenter
from src.extraction.experience import ExperienceExtractor

def test_three_distinct_experiences_do_not_merge():
    """Verify that 3 sequential jobs with roles, companies, and date ranges segment into exactly 3 episodes."""
    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text="Data Analyst Intern\nComplyHappy Pvt Ltd.\n19 Jun 2023 – 11 Aug 2023\nDeveloped automated reporting scripts.",
            is_bold=True,
            section="experience"
        ),
        DocumentBlock(
            id="b2",
            text="• Built exploratory data analysis pipelines in Python and Pandas.\n• Delivered weekly KPI metrics to executives.",
            section="experience"
        ),
        DocumentBlock(
            id="b3",
            text="Machine Learning Engineer Intern\nByzWiz Technologies\n12 Jun 2024 – 07 Aug 2024\nImplemented credit risk classification models.",
            is_bold=True,
            section="experience"
        ),
        DocumentBlock(
            id="b4",
            text="• Trained Scikit-Learn Random Forest and XGBoost classifiers.\n• Achieved 91% precision on test validation sets.",
            section="experience"
        ),
        DocumentBlock(
            id="b5",
            text="AI Engineer Intern\nMarva AI\n06 Jan 2025 – 17 May 2025\nArchitected LLM agent system.",
            is_bold=True,
            section="experience"
        ),
        DocumentBlock(
            id="b6",
            text="• Integrated LangChain retrieval augmented generation pipelines.\n• Containerized microservices using Docker.",
            section="experience"
        )
    ]

    episodes = segmenter.segment_experiences(blocks)

    # STRICT ASSERTION: Must be exactly 3 discrete episodes!
    assert len(episodes) == 3, f"Expected 3 separate experiences, but got {len(episodes)}"

    # Check Episode 1
    assert "Analyst" in episodes[0].role
    assert "ComplyHappy" in episodes[0].company
    assert episodes[0].start_date is not None
    assert len(episodes[0].bullet_points) >= 1

    # Check Episode 2
    assert "Machine Learning" in episodes[1].role
    assert "ByzWiz" in episodes[1].company
    assert "2024" in episodes[1].start_date
    assert len(episodes[1].bullet_points) >= 1

    # Check Episode 3
    assert "AI Engineer" in episodes[2].role
    assert "Marva AI" in episodes[2].company
    assert "2025" in episodes[2].start_date
    assert len(episodes[2].bullet_points) >= 1

def test_experience_without_explicit_end_date():
    """Verify handling of ongoing/Present roles."""
    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text="Senior Data Scientist\nTechCorp Global\nJan 2023 – Present\nLeading the recommendation systems squad.",
            is_bold=True,
            section="experience"
        ),
        DocumentBlock(
            id="b2",
            text="• Scaled PyTorch neural recommendation models.\n• Optimized inference latency by 40%.",
            section="experience"
        )
    ]

    episodes = segmenter.segment_experiences(blocks)
    assert len(episodes) == 1
    assert episodes[0].is_current is True
    assert episodes[0].end_date == "Present"
def test_role_company_without_legal_suffix():

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "AI Engineer Intern\n"
                "Marva AI\n"
                "06 Jan 2025 – 17 May 2025\n"
                "Architected LLM agent system."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1
    assert episodes[0].role == "AI Engineer Intern"
    assert episodes[0].company == "Marva AI"
    assert episodes[0].start_date == "06 Jan 2025"
    assert episodes[0].end_date == "17 May 2025"


def test_same_line_role_at_company():

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Machine Learning Engineer @ OpenAI\n"
                "2024 – Present\n"
                "Built production inference systems."
            ),
            section="experience",
        )
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1
    assert "Machine Learning Engineer" in episodes[0].role
    assert "OpenAI" in episodes[0].company
    assert episodes[0].is_current is True


def test_description_sentences_do_not_create_experiences():

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "I developed machine learning models.\n"
                "We built data pipelines.\n"
                "Built several data science projects."
            ),
            section="experience",
        )
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) <= 1


def test_research_experience():

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Research Assistant\n"
                "Stanford University\n"
                "2023 – 2024\n"
                "Conducted research on transformer architectures."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1
    assert "Research Assistant" in episodes[0].role
    assert "Stanford University" in episodes[0].company


def test_current_experience_without_end_date():

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text=(
                "Senior Data Scientist\n"
                "Nexus Analytics\n"
                "Jan 2023 – Present\n"
                "Leading recommendation systems."
            ),
            is_bold=True,
            section="experience",
        )
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1
    assert episodes[0].is_current is True
    assert episodes[0].end_date == "Present"

def test_role_company_date_in_separate_blocks():
    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text="Data Analyst Intern",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b2",
            text="ComplyHappy Pvt Ltd",
            section="experience",
        ),
        DocumentBlock(
            id="b3",
            text="19 Jun 2023 – 11 Aug 2023",
            section="experience",
        ),
        DocumentBlock(
            id="b4",
            text="Developed automated reporting pipelines using Python.",
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1
    assert episodes[0].role == "Data Analyst Intern"
    assert "ComplyHappy" in episodes[0].company
    assert episodes[0].start_date is not None
    assert episodes[0].end_date is not None
    assert len(episodes[0].bullet_points) >= 1

def test_hyphenated_role_domain_is_not_company():
    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text="Senior Consultant - Data Science",
            is_bold=True,
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) <= 1

    if episodes:
        assert episodes[0].role == "Senior Consultant - Data Science"
        assert episodes[0].company != "Data Science"

def test_location_is_not_company():
    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text="Data Analyst Intern",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b2",
            text="Pune, India",
            section="experience",
        ),
        DocumentBlock(
            id="b3",
            text="Jun 2023 – Aug 2023",
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1
    assert episodes[0].company != "Pune, India"
    assert episodes[0].location == "Pune, India"


def test_experience_without_dates():
    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text="Software Engineer Intern",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b2",
            text="Nexus Analytics",
            section="experience",
        ),
        DocumentBlock(
            id="b3",
            text="Developed backend APIs and data processing services.",
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1
    assert episodes[0].role == "Software Engineer Intern"
    assert episodes[0].company == "Nexus Analytics"
    assert episodes[0].start_date is None


def test_short_non_bold_description_is_preserved():
    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="b1",
            text="AI Engineer Intern\nMarva AI",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="b2",
            text="Backend Development",
            is_bold=False,
            section="experience",
        ),
        DocumentBlock(
            id="b3",
            text="Model Deployment",
            is_bold=False,
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1

    assert "Backend Development" in episodes[0].bullet_points
    assert "Model Deployment" in episodes[0].bullet_points


def test_same_company_different_roles_are_separate_experiences():
    """Verify promotions/role changes at the same company remain separate."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="same_company_1",
            text=(
                "Software Engineer\n"
                "TechCorp\n"
                "Jan 2022 – Dec 2023\n"
                "Built backend services and APIs."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="same_company_2",
            text=(
                "Senior Software Engineer\n"
                "TechCorp\n"
                "Jan 2024 – Present\n"
                "Led backend architecture and engineering teams."
            ),
            is_bold=True,
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 2

    # Experience 1
    assert episodes[0].role == "Software Engineer"
    assert episodes[0].company == "TechCorp"
    assert episodes[0].start_date == "Jan 2022"
    assert episodes[0].end_date == "Dec 2023"

    # Experience 2
    assert episodes[1].role == "Senior Software Engineer"
    assert episodes[1].company == "TechCorp"
    assert episodes[1].start_date == "Jan 2024"
    assert episodes[1].is_current is True
    assert episodes[1].end_date == "Present"

    # Both experiences must remain independent.
    assert episodes[0].id != episodes[1].id                

def test_same_role_same_company_different_periods_are_separate_experiences():
    """Verify repeated employment at the same company remains separate."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="repeat_1",
            text=(
                "Data Analyst\n"
                "ABC Corp\n"
                "Jan 2021 – Dec 2021\n"
                "Built reporting dashboards and automated data workflows."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="repeat_2",
            text=(
                "Data Analyst\n"
                "ABC Corp\n"
                "Jun 2023 – Aug 2023\n"
                "Developed analytics pipelines for business reporting."
            ),
            is_bold=True,
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 2

    assert episodes[0].role == "Data Analyst"
    assert episodes[0].company == "ABC Corp"
    assert episodes[0].start_date == "Jan 2021"
    assert episodes[0].end_date == "Dec 2021"

    assert episodes[1].role == "Data Analyst"
    assert episodes[1].company == "ABC Corp"
    assert episodes[1].start_date == "Jun 2023"
    assert episodes[1].end_date == "Aug 2023"

    assert episodes[0].id != episodes[1].id                


def test_dated_experience_followed_by_dated_less_experience_stays_separate():
    """Verify a new role/company without dates does not merge into the previous job."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="partial_1",
            text=(
                "Data Analyst\n"
                "ABC Corp\n"
                "Jan 2023 – Jun 2023\n"
                "Built automated reporting dashboards."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="partial_2",
            text=(
                "Machine Learning Intern\n"
                "XYZ Labs"
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="partial_3",
            text=(
                "Developed machine learning models and "
                "data preprocessing pipelines."
            ),
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 2

    # ---------------------------------------------------------
    # Experience 1
    # ---------------------------------------------------------

    assert episodes[0].role == "Data Analyst"
    assert episodes[0].company == "ABC Corp"
    assert episodes[0].start_date == "Jan 2023"
    assert episodes[0].end_date == "Jun 2023"

    # ---------------------------------------------------------
    # Experience 2
    # ---------------------------------------------------------

    assert episodes[1].role == "Machine Learning Intern"
    assert episodes[1].company == "XYZ Labs"

    # Date is intentionally missing.
    assert episodes[1].start_date is None
    assert episodes[1].end_date is None

    assert any(
        "machine learning models" in bullet.lower()
        for bullet in episodes[1].bullet_points
    )

    # ---------------------------------------------------------
    # They must remain independent.
    # ---------------------------------------------------------

    assert episodes[0].id != episodes[1].id

def test_role_company_date_and_description_in_separate_blocks():
    """Verify a single experience can be assembled from separate blocks."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="split_1",
            text="Data Analyst Intern",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="split_2",
            text="ComplyHappy Pvt Ltd",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="split_3",
            text="19 Jun 2023 – 11 Aug 2023",
            is_bold=False,
            section="experience",
        ),
        DocumentBlock(
            id="split_4",
            text="Built automated reporting dashboards.",
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1

    episode = episodes[0]

    assert episode.role == "Data Analyst Intern"
    assert "ComplyHappy" in episode.company
    assert episode.start_date == "19 Jun 2023"
    assert episode.end_date == "11 Aug 2023"

    assert (
        "Built automated reporting dashboards."
        in episode.bullet_points
    )        

def test_multiple_sequential_dated_and_datedless_experiences_stay_separate():
    """Verify multiple date-less experiences do not merge with each other."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="multi_1",
            text=(
                "Data Analyst\n"
                "ABC Corp\n"
                "Jan 2023 – Jun 2023\n"
                "Built reporting dashboards."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="multi_2",
            text=(
                "Machine Learning Intern\n"
                "XYZ Labs"
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="multi_3",
            text=(
                "Developed machine learning models."
            ),
            section="experience",
        ),
        DocumentBlock(
            id="multi_4",
            text=(
                "AI Engineer Intern\n"
                "Marva AI"
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="multi_5",
            text=(
                "Built LLM-based agent systems."
            ),
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 3

    # ---------------------------------------------------------
    # Experience 1
    # ---------------------------------------------------------

    assert episodes[0].role == "Data Analyst"
    assert episodes[0].company == "ABC Corp"
    assert episodes[0].start_date == "Jan 2023"
    assert episodes[0].end_date == "Jun 2023"

    # ---------------------------------------------------------
    # Experience 2
    # ---------------------------------------------------------

    assert episodes[1].role == "Machine Learning Intern"
    assert episodes[1].company == "XYZ Labs"
    assert episodes[1].start_date is None

    assert (
        "Developed machine learning models."
        in episodes[1].bullet_points
    )

    # ---------------------------------------------------------
    # Experience 3
    # ---------------------------------------------------------

    assert episodes[2].role == "AI Engineer Intern"
    assert episodes[2].company == "Marva AI"
    assert episodes[2].start_date is None

    assert (
        "Built LLM-based agent systems."
        in episodes[2].bullet_points
    )    

def test_company_before_role_is_single_experience():
    """Verify company-before-role ordering is handled as one experience."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="company_first_1",
            text="ComplyHappy Pvt Ltd",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="company_first_2",
            text="Data Analyst Intern",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="company_first_3",
            text="19 Jun 2023 – 11 Aug 2023",
            is_bold=False,
            section="experience",
        ),
        DocumentBlock(
            id="company_first_4",
            text="Built automated reporting dashboards.",
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1

    episode = episodes[0]

    assert episode.role == "Data Analyst Intern"
    assert "ComplyHappy" in episode.company
    assert episode.start_date == "19 Jun 2023"
    assert episode.end_date == "11 Aug 2023"

    assert (
        "Built automated reporting dashboards."
        in episode.bullet_points
    )

def test_non_bold_company_like_description_does_not_create_experience():
    """Verify company-like descriptive text does not create a new episode."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="company_desc_1",
            text=(
                "AI Engineer Intern\n"
                "Marva AI\n"
                "Developed backend services."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="company_desc_2",
            text="OpenAI Integration",
            is_bold=False,
            section="experience",
        ),
        DocumentBlock(
            id="company_desc_3",
            text="Google Cloud Deployment",
            is_bold=False,
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1

    assert episodes[0].role == "AI Engineer Intern"
    assert episodes[0].company == "Marva AI"

    assert "OpenAI Integration" in episodes[0].bullet_points
    assert "Google Cloud Deployment" in episodes[0].bullet_points

def test_dated_then_datedless_experience_description_attaches_to_new_experience():
    """Verify description after a new date-less experience stays with that experience."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="boundary_1",
            text=(
                "Data Analyst\n"
                "ABC Corp\n"
                "Jan 2023 – Jun 2023\n"
                "Built automated dashboards."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="boundary_2",
            text=(
                "Machine Learning Intern\n"
                "XYZ Labs"
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="boundary_3",
            text=(
                "Developed machine learning models and "
                "data preprocessing pipelines."
            ),
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 2

    # Previous experience
    assert episodes[0].role == "Data Analyst"
    assert episodes[0].company == "ABC Corp"

    # New date-less experience
    assert episodes[1].role == "Machine Learning Intern"
    assert episodes[1].company == "XYZ Labs"

    assert (
        "Developed machine learning models and "
        "data preprocessing pipelines."
        in episodes[1].bullet_points
    )

    # Description must NOT leak backwards.
    assert (
        "Developed machine learning models and "
        "data preprocessing pipelines."
        not in episodes[0].bullet_points
    )

def test_role_before_company_without_date_is_single_experience():
    """Verify role → company without a date creates one experience."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="role_first_1",
            text="Data Science Intern",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="role_first_2",
            text="ABC Technologies",
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="role_first_3",
            text=(
                "Developed machine learning models and "
                "automated data preprocessing pipelines."
            ),
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 1

    assert episodes[0].role == "Data Science Intern"
    assert episodes[0].company == "ABC Technologies"

    assert (
        "Developed machine learning models and "
        "automated data preprocessing pipelines."
        in episodes[0].bullet_points
    )

    assert episodes[0].start_date is None
    assert episodes[0].end_date is None    

def test_same_role_company_repeated_with_separate_periods_and_descriptions():
    """Verify repeated role/company entries with different periods stay separate."""

    segmenter = ExperienceSegmenter()

    blocks = [
        DocumentBlock(
            id="repeat_1",
            text=(
                "Software Engineer\n"
                "ABC Technologies\n"
                "Jan 2022 – Jun 2022\n"
                "Built backend APIs and data services."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="repeat_2",
            text=(
                "Software Engineer\n"
                "ABC Technologies\n"
                "Jan 2024 – Jun 2024\n"
                "Designed machine learning infrastructure."
            ),
            is_bold=True,
            section="experience",
        ),
    ]

    episodes = segmenter.segment_experiences(blocks)

    assert len(episodes) == 2

    # First period
    assert episodes[0].role == "Software Engineer"
    assert episodes[0].company == "ABC Technologies"
    assert episodes[0].start_date == "Jan 2022"
    assert episodes[0].end_date == "Jun 2022"
    assert (
        "Built backend APIs and data services."
        in episodes[0].bullet_points
    )

    # Second period
    assert episodes[1].role == "Software Engineer"
    assert episodes[1].company == "ABC Technologies"
    assert episodes[1].start_date == "Jan 2024"
    assert episodes[1].end_date == "Jun 2024"
    assert (
        "Designed machine learning infrastructure."
        in episodes[1].bullet_points
    )

def test_extractor_preserves_same_role_company_different_periods():
    """Verify the public extractor never merges repeated role/company periods."""

    extractor = ExperienceExtractor()

    blocks = [
        DocumentBlock(
            id="repeat_1",
            text=(
                "Software Engineer\n"
                "ABC Technologies\n"
                "Jan 2022 – Jun 2022\n"
                "Built backend APIs and data services."
            ),
            is_bold=True,
            section="experience",
        ),
        DocumentBlock(
            id="repeat_2",
            text=(
                "Software Engineer\n"
                "ABC Technologies\n"
                "Jan 2024 – Jun 2024\n"
                "Designed machine learning infrastructure."
            ),
            is_bold=True,
            section="experience",
        ),
    ]

    results = extractor.extract(blocks)

    assert len(results) == 2

    # ---------------------------------------------------------
    # First experience
    # ---------------------------------------------------------

    assert results[0].role == "Software Engineer"
    assert results[0].company == "ABC Technologies"
    assert results[0].start_date == "Jan 2022"
    assert results[0].end_date == "Jun 2022"

    assert (
        "Built backend APIs and data services."
        in results[0].bullet_points
    )

    # ---------------------------------------------------------
    # Second experience
    # ---------------------------------------------------------

    assert results[1].role == "Software Engineer"
    assert results[1].company == "ABC Technologies"
    assert results[1].start_date == "Jan 2024"
    assert results[1].end_date == "Jun 2024"

    assert (
        "Designed machine learning infrastructure."
        in results[1].bullet_points
    )        