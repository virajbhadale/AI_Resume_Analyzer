"""Dynamic professional title and domain detection for resumes."""

import re
from typing import List, Tuple


class ResumeProfileExtractor:
    """
    Detects a candidate's professional title and domain from resume text.

    Important:
    - Does NOT default to Data Scientist.
    - Does NOT inject technical skills.
    - Uses evidence from the resume itself.
    """

    DOMAIN_KEYWORDS = {
        "Data Science": [
            "data scientist",
            "data science",
            "machine learning",
            "predictive modeling",
            "statistical modeling",
            "deep learning",
            "pandas",
            "scikit-learn",
            "tensorflow",
            "pytorch",
        ],
        "Artificial Intelligence": [
            "artificial intelligence",
            "ai engineer",
            "ai engineering",
            "computer vision",
            "natural language processing",
            "nlp",
            "generative ai",
            "llm",
            "large language model",
        ],
        "Software Engineering": [
            "software engineer",
            "software developer",
            "full stack developer",
            "backend developer",
            "frontend developer",
            "web developer",
            "application developer",
        ],
        "Data Engineering": [
            "data engineer",
            "data engineering",
            "etl",
            "data pipeline",
            "apache spark",
            "hadoop",
            "data warehouse",
        ],
        "Data Analytics": [
            "data analyst",
            "business analyst",
            "data analytics",
            "business intelligence",
            "tableau",
            "power bi",
        ],
        "Marketing": [
            "marketing",
            "digital marketing",
            "marketing manager",
            "marketing executive",
            "seo",
            "sem",
            "content marketing",
            "brand management",
            "social media marketing",
        ],
        "Finance": [
            "finance",
            "financial analyst",
            "financial analysis",
            "investment",
            "accounting",
            "accountant",
            "financial planning",
            "portfolio management",
        ],
        "Human Resources": [
            "human resources",
            "hr manager",
            "hr executive",
            "recruiter",
            "recruitment",
            "talent acquisition",
            "people operations",
        ],
        "Sales": [
            "sales",
            "sales manager",
            "sales executive",
            "business development",
            "account management",
            "client acquisition",
        ],
        "Operations": [
            "operations",
            "operations manager",
            "supply chain",
            "logistics",
            "procurement",
        ],
    }

    TITLE_PATTERNS = [
        r"\b(?:senior|junior|lead|associate|assistant|chief|principal)?\s*"
        r"(?:data scientist|data analyst|data engineer|machine learning engineer|"
        r"ai engineer|software engineer|software developer|full stack developer|"
        r"backend developer|frontend developer|web developer|"
        r"marketing manager|marketing executive|digital marketing specialist|"
        r"financial analyst|accountant|hr manager|hr executive|recruiter|"
        r"sales manager|sales executive|business development manager|"
        r"operations manager)\b",
    ]

    def extract(
        self,
        summary: str,
        experience_roles: List[str],
        skills: List[str],
        full_text: str = "",
    ) -> Tuple[str, str]:

        combined_text = " ".join(
            [
                summary or "",
                " ".join(experience_roles),
                " ".join(skills),
                full_text or "",
            ]
        )

        text_lower = combined_text.lower()

        current_title = self._extract_title(
            summary=summary,
            experience_roles=experience_roles,
            full_text=full_text,
        )

        domain_category = self._detect_domain(text_lower)

        return current_title, domain_category

    def _extract_title(
        self,
        summary: str,
        experience_roles: List[str],
        full_text: str,
    ) -> str:

        # 1. Prefer explicit role from experience
        for role in experience_roles:
            role_clean = role.strip()
            if role_clean and len(role_clean.split()) <= 8:
                return role_clean

        # 2. Search summary
        if summary:
            for pattern in self.TITLE_PATTERNS:
                match = re.search(pattern, summary, re.IGNORECASE)
                if match:
                    return match.group(0).strip().title()

        # 3. Search full resume text
        if full_text:
            for pattern in self.TITLE_PATTERNS:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    return match.group(0).strip().title()

        return "Professional"

    def _detect_domain(self, text_lower: str) -> str:
        scores = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = 0

            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1

            if score > 0:
                scores[domain] = score

        if not scores:
            return "General"

        # Highest evidence score wins.
        return max(scores, key=scores.get)