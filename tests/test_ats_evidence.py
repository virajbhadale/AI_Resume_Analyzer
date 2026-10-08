from src.ats.analyzer import ATSAnalyzer
from src.document_models import (
    CanonicalResume,
    ContactInfo,
    ExperienceEpisode,
    ProjectEntry,
)


def test_ats_collects_structured_experience_evidence():

    resume = CanonicalResume(
        contact=ContactInfo(
            email="test@example.com"
        ),

        current_title="AI Engineer",

        domain_category="Artificial Intelligence",

        summary=(
            "AI Engineer specializing in "
            "machine learning systems."
        ),

        skills=[
            "Python",
        ],

        experience=[
            ExperienceEpisode(
                id="exp_1",
                role="AI Engineer",
                company="Example AI",
                description=(
                    "Built production machine "
                    "learning services."
                ),
                bullet_points=[
                    "Developed RAG pipelines "
                    "for enterprise search."
                ],
                technologies_mentioned=[
                    "LangChain",
                    "FastAPI",
                    "RAG",
                ],
            )
        ],

        projects=[
            ProjectEntry(
                id="project_1",
                title="RAG Assistant",
                description=(
                    "Enterprise document assistant."
                ),
                technologies=[
                    "LangGraph",
                    "ChromaDB",
                ],
            )
        ],
    )

    analyzer = ATSAnalyzer()

    corpus = analyzer._build_evidence_corpus(
        resume
    )

    corpus_text = "\n".join(
        text
        for _, text in corpus
    )

    assert "LangChain" in corpus_text
    assert "FastAPI" in corpus_text
    assert "RAG" in corpus_text
    assert "LangGraph" in corpus_text
    assert "ChromaDB" in corpus_text


def test_ats_evidence_returns_empty_for_missing_skill():

    resume = CanonicalResume(
        contact=ContactInfo(
            email="test@example.com"
        ),
        current_title="AI Engineer",
        skills=[
            "Python",
        ],
    )

    analyzer = ATSAnalyzer()

    corpus = analyzer._build_evidence_corpus(
        resume
    )

    evidence = analyzer._find_skill_evidence(
        "Kubernetes",
        corpus,
    )

    assert evidence == []