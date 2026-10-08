"""
Universal experience fixtures.

These fixtures describe structural resume patterns rather than
specific real resumes. They are intentionally small so a failing
case points to a specific architectural behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperienceExpectation:
    role: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: Optional[bool] = None


@dataclass(frozen=True)
class ExperienceCase:
    case_id: str
    description: str
    blocks: tuple[str, ...]
    expected: tuple[ExperienceExpectation, ...]
    category: str


UNIVERSAL_EXPERIENCE_CASES = (
    ExperienceCase(
        case_id="basic_role_company_date",
        description="Standard role, company and date sequence.",
        blocks=(
            "Senior Data Scientist",
            "ABC Technologies",
            "Jan 2024 - Present",
        ),
        expected=(
            ExperienceExpectation(
                role="Senior Data Scientist",
                company="ABC Technologies",
                start_date="Jan 2024",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="basic",
    ),

    ExperienceCase(
        case_id="role_company_same_line",
        description="Role and company are on one line.",
        blocks=(
            "Machine Learning Engineer @ OpenAI",
            "2024 - Present",
        ),
        expected=(
            ExperienceExpectation(
                role="Machine Learning Engineer",
                company="OpenAI",
                start_date="2024",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="basic",
    ),

    ExperienceCase(
        case_id="role_split_across_lines",
        description="A role is split across multiple lines.",
        blocks=(
            "Senior Machine Learning",
            "Engineer",
            "ABC Technologies",
            "2022 - Present",
        ),
        expected=(
            ExperienceExpectation(
                role="Senior Machine Learning Engineer",
                company="ABC Technologies",
                start_date="2022",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="multi_block",
    ),

    ExperienceCase(
        case_id="company_before_role",
        description="Company appears before the role.",
        blocks=(
            "ABC Technologies",
            "Senior Data Scientist",
            "Jan 2022 - Present",
        ),
        expected=(
            ExperienceExpectation(
                role="Senior Data Scientist",
                company="ABC Technologies",
                start_date="Jan 2022",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="ordering",
    ),

    ExperienceCase(
        case_id="missing_company",
        description="Experience has no explicit company.",
        blocks=(
            "Data Scientist",
            "Jan 2022 - Present",
            "Built predictive models.",
        ),
        expected=(
            ExperienceExpectation(
                role="Data Scientist",
                company=None,
                start_date="Jan 2022",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="missing_metadata",
    ),

    ExperienceCase(
        case_id="missing_role",
        description="Experience has company and dates but no explicit role.",
        blocks=(
            "ABC Technologies",
            "Jan 2022 - Present",
            "Developed machine learning systems.",
        ),
        expected=(
            ExperienceExpectation(
                role=None,
                company="ABC Technologies",
                start_date="Jan 2022",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="missing_metadata",
    ),

    ExperienceCase(
        case_id="missing_dates",
        description="Experience has role and company but no dates.",
        blocks=(
            "Data Scientist",
            "ABC Technologies",
            "Developed forecasting models.",
        ),
        expected=(
            ExperienceExpectation(
                role="Data Scientist",
                company="ABC Technologies",
                start_date=None,
                end_date=None,
            ),
        ),
        category="missing_metadata",
    ),

    ExperienceCase(
        case_id="same_company_multiple_roles",
        description="Two roles at the same company must remain separate.",
        blocks=(
            "Machine Learning Engineer",
            "ABC Technologies",
            "Jan 2022 - Dec 2023",
            "Built ML systems.",
            "Senior Machine Learning Engineer",
            "ABC Technologies",
            "Jan 2024 - Present",
            "Led ML platform development.",
        ),
        expected=(
            ExperienceExpectation(
                role="Machine Learning Engineer",
                company="ABC Technologies",
                start_date="Jan 2022",
                end_date="Dec 2023",
            ),
            ExperienceExpectation(
                role="Senior Machine Learning Engineer",
                company="ABC Technologies",
                start_date="Jan 2024",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="boundaries",
    ),

    ExperienceCase(
        case_id="same_role_multiple_companies",
        description="The same role at different companies must remain separate.",
        blocks=(
            "Data Scientist",
            "Company A",
            "Jan 2020 - Dec 2021",
            "Data Scientist",
            "Company B",
            "Jan 2022 - Present",
        ),
        expected=(
            ExperienceExpectation(
                role="Data Scientist",
                company="Company A",
                start_date="Jan 2020",
                end_date="Dec 2021",
            ),
            ExperienceExpectation(
                role="Data Scientist",
                company="Company B",
                start_date="Jan 2022",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="boundaries",
    ),

    ExperienceCase(
        case_id="location_not_company",
        description="A standalone location must not become a company.",
        blocks=(
            "Data Scientist",
            "Pune, India",
            "Jan 2022 - Present",
        ),
        expected=(
            ExperienceExpectation(
                role="Data Scientist",
                company=None,
                location="Pune, India",
                start_date="Jan 2022",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="negative",
    ),

    ExperienceCase(
        case_id="prose_not_company",
        description="Ordinary prose must not become company evidence.",
        blocks=(
            "Data Scientist",
            "Jan 2022 - Present",
            "Presented findings at weekly research meetings.",
        ),
        expected=(
            ExperienceExpectation(
                role="Data Scientist",
                company=None,
                start_date="Jan 2022",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="negative",
    ),

    ExperienceCase(
        case_id="experience_with_bullets",
        description="Description bullets belong to the preceding episode.",
        blocks=(
            "Machine Learning Engineer",
            "ABC Technologies",
            "2022 - Present",
            "- Built production ML pipelines.",
            "- Improved model latency.",
        ),
        expected=(
            ExperienceExpectation(
                role="Machine Learning Engineer",
                company="ABC Technologies",
                start_date="2022",
                end_date="Present",
                is_current=True,
            ),
        ),
        category="multi_block",
    ),
)