"""Taxonomy-backed Technical Skills Extractor."""
import re
import os
import yaml
from typing import List, Dict, Set, Tuple
from src.document_models import DocumentBlock

DEFAULT_SKILLS = [
    # Programming Languages
    "Python", "SQL", "R", "C++", "Java", "JavaScript", "TypeScript", "Bash", "Scala", "Go",
    # ML / AI / Data Science
    "Machine Learning", "Deep Learning", "Natural Language Processing", "NLP", "Computer Vision",
    "Scikit-Learn", "PyTorch", "TensorFlow", "Keras", "Pandas", "NumPy", "SciPy",
    "Feature Engineering", "Model Evaluation", "XGBoost", "LightGBM", "CatBoost",
    # GenAI / LLM
    "Generative AI", "LLMs", "RAG", "LangChain", "LangGraph", "LlamaIndex",
    "Transformers", "Hugging Face", "Vector Databases", "Embeddings", "Prompt Engineering",
    "Pinecone", "ChromaDB", "Qdrant", "Weaviate", "vLLM", "Ollama",
    # Data Engineering & DBs
    "ETL", "Data Pipelines", "Apache Spark", "Spark", "PySpark", "Airflow", "Kafka",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Snowflake", "BigQuery", "Databricks", "dbt",
    # MLOps & Cloud
    "Docker", "Kubernetes", "MLflow", "CI/CD", "Git", "GitHub Actions", "Linux",
    "AWS", "GCP", "Azure", "SageMaker", "Vertex AI",
    "Model Deployment", "Model Serving", "APIs",
    # Analytics & BI
    "Tableau", "Power BI", "Excel", "Data Visualization", "Matplotlib", "Seaborn",
    "Statistics", "Hypothesis Testing", "A/B Testing", "FastAPI", "Flask", "Django"
]


# Common spelling/abbreviation variants found across technical resumes.
# Values are canonical names used by ATS/JD matching.
SKILL_ALIASES = {
    "sklearn": "Scikit-Learn",
    "scikit learn": "Scikit-Learn",
    "scikit-learning": "Scikit-Learn",
    "large language model": "LLMs",
    "large language models": "LLMs",
    "llm": "LLMs",
    "llms": "LLMs",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "retrieval augmented generations": "RAG",
    "lang chain": "LangChain",
    "lang graph": "LangGraph",
    "vector db": "Vector Databases",
    "vector database": "Vector Databases",
    "vector databases": "Vector Databases",
    "vector store": "Vector Databases",
    "vector stores": "Vector Databases",
    "embedding": "Embeddings",
    "embeddings": "Embeddings",
    "prompt engineering": "Prompt Engineering",
    "natural language processing": "NLP",
    "nlp": "NLP",
    "computer vision": "Computer Vision",
    "powerbi": "Power BI",
    "power-bi": "Power BI",
    "ms power bi": "Power BI",
    "postgre sql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongo db": "MongoDB",
    "apache spark": "Spark",
    "pyspark": "PySpark",
    "machine-learning": "Machine Learning",
    "deep-learning": "Deep Learning",
    "generative-ai": "Generative AI",
    "genai": "Generative AI",
    "ci cd": "CI/CD",
    "continuous integration/continuous deployment": "CI/CD",
    "rest api": "APIs",
    "rest apis": "APIs",
    "restful api": "APIs",
    "restful apis": "APIs",
    "model deployment": "Model Deployment",
    "model serving": "Model Serving",
    "api": "APIs",
    "apis": "APIs",
    "rest api": "APIs",
    "rest apis": "APIs",
    "restful api": "APIs",
    "restful apis": "APIs",
    "web api": "APIs",
    "web apis": "APIs",
}

class SkillExtractor:
    def __init__(self, roles_config_path: str = "config/job_roles.yaml"):
        self.skills_pool: Set[str] = set(DEFAULT_SKILLS)
        self._load_from_config(roles_config_path)

        # Normalize role/config skill names to one canonical vocabulary.
        # This prevents duplicates such as NLP/Natural Language Processing
        # or APIs/RESTful APIs from appearing as separate competencies.
        self.skills_pool = {
            SKILL_ALIASES.get(str(skill).strip().lower(), str(skill).strip())
            for skill in self.skills_pool
            if str(skill).strip()
        }
        self.skills_pool.update(SKILL_ALIASES.values())
        self._build_regex_cache()

    def _load_from_config(self, config_path: str):
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for role_info in data.get("roles", {}).values():
                    for sk in role_info.get("core_skills", []):
                        self.skills_pool.add(sk)
                    for sk in role_info.get("secondary_skills", []):
                        self.skills_pool.add(sk)

    def _build_regex_cache(self):
        """Compile canonical skills and aliases, longest phrases first."""
        patterns: List[Tuple[str, str]] = []

        for canonical in self.skills_pool:
            patterns.append((canonical, canonical))

        for alias, canonical in SKILL_ALIASES.items():
            patterns.append((canonical, alias))

        patterns.sort(key=lambda item: len(item[1]), reverse=True)

        self.compiled_skills: List[Tuple[str, re.Pattern]] = []
        for canonical_name, surface_form in patterns:
            escaped = re.escape(surface_form)
            pattern = re.compile(
                rf"(?<![a-zA-Z0-9_#+]){escaped}(?![a-zA-Z0-9_#+])",
                re.IGNORECASE,
            )
            self.compiled_skills.append((canonical_name, pattern))

    def extract_from_text(self, text: str) -> List[str]:
        """Extract normalized technical skills from arbitrary resume text."""
        if not text:
            return []

        found: Set[str] = set()
        for canonical_name, pattern in self.compiled_skills:
            if pattern.search(text):
                found.add(canonical_name)

        return sorted(found)

    def extract(self, blocks: List[DocumentBlock]) -> List[str]:
        """Extract normalized skills from dedicated and contextual resume content."""
        # Dedicated skills sections get explicit priority, but universal
        # resumes often omit that section and mention technologies in
        # projects, experience, education, certifications, or summaries.
        skills_text = "\n".join(
            b.text for b in blocks if b.section == "skills" and not b.is_heading
        )
        section_skills = self.extract_from_text(skills_text)

        contextual_sections = {
            "summary",
            "experience",
            "projects",
            "education",
            "certifications",
            "research",
            "publications",
            "achievements",
        }
        contextual_text = "\n".join(
            b.text
            for b in blocks
            if b.section in contextual_sections and not b.is_heading
        )
        contextual_skills = self.extract_from_text(contextual_text)

        # As a final structure-agnostic fallback, scan all content except
        # obvious personal-contact/administrative sections. The taxonomy is
        # constrained, so this does not turn arbitrary prose into skills.
        fallback_text = "\n".join(
            b.text
            for b in blocks
            if b.section not in {"contact", "references", "languages", "interests"}
            and not b.is_heading
        )
        fallback_skills = self.extract_from_text(fallback_text)

        return sorted(set(section_skills + contextual_skills + fallback_skills))
