"""Main Pipeline Orchestrator for AI Resume Analyser V2."""
import os
import json
from typing import Dict, Any, Optional, List

from src.parser import DocumentParser
from src.preprocessing.unicode_normalizer import normalize_unicode
from src.layout import LayoutEngine
from src.cleaner import DocumentCleaner
from src.section_extractor import SectionExtractor
from src.extraction.resume_profile import ResumeProfileExtractor
from src.extraction.contact import ContactExtractor
from src.extraction.skills import SkillExtractor
from src.extraction.education import EducationExtractor
from src.extraction.projects import ProjectExtractor
from src.extraction.certifications import CertificationExtractor
from src.extraction.experience import ExperienceExtractor
from src.inference.confidence import ConfidenceScorer
from src.inference.graph import EpisodeGraphBuilder
from src.ats.analyzer import ATSAnalyzer
from src.ats.jd_matcher import JobDescriptionMatcher
from src.inference.resume_builder import ResumeBuilder

class ResumeAnalysisPipeline:
    def __init__(self, sections_config: str = "config/sections.yaml", roles_config: str = "config/job_roles.yaml"):
        self.parser = DocumentParser()
        self.layout_engine = LayoutEngine()
        self.cleaner = DocumentCleaner()
        self.section_extractor = SectionExtractor(sections_config)
        self.contact_extractor = ContactExtractor()
        self.skill_extractor = SkillExtractor(roles_config)
        self.education_extractor = EducationExtractor()
        self.project_extractor = ProjectExtractor(self.skill_extractor)
        self.cert_extractor = CertificationExtractor()
        self.experience_extractor = ExperienceExtractor(self.skill_extractor)
        self.resume_profile_extractor = ResumeProfileExtractor()
        self.confidence_scorer = ConfidenceScorer()
        self.graph_builder = EpisodeGraphBuilder()
        self.resume_builder = ResumeBuilder()
        self.ats_analyzer = ATSAnalyzer(roles_config)
        self.jd_matcher = JobDescriptionMatcher(self.skill_extractor)

    def analyze_resume(
        self,
        file_path: str,
        target_role: Optional[str] = None,
        jd_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes full multi-stage layout-aware analysis on a PDF/DOCX file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. Document Ingestion
        raw_blocks = self.parser.parse(file_path)

        # 1.1 Unicode Normalization
        for block in raw_blocks:
            block.text = normalize_unicode(block.text)

        # 2. Layout Reconstruction & Column Ordering
        ordered_blocks = self.layout_engine.assign_columns_and_reading_order(raw_blocks)

        # 3. Document Cleaning
        cleaned_blocks = self.cleaner.filter_blocks(ordered_blocks)

        # 4. Section Boundary Tagging
        section_tagged_blocks = self.section_extractor.segment_document_by_sections(cleaned_blocks)

        # 5. Entity Extractions
        contact_info = self.contact_extractor.extract(section_tagged_blocks)
        extracted_skills = self.skill_extractor.extract(section_tagged_blocks)
        education_entries = self.education_extractor.extract(section_tagged_blocks)
        experience_episodes = self.experience_extractor.extract(section_tagged_blocks)
        project_entries = self.project_extractor.extract(section_tagged_blocks)
        cert_entries = self.cert_extractor.extract(section_tagged_blocks)

        # Summary extraction (if summary section exists)
        summary_blocks = [
            b for b in section_tagged_blocks
            if b.section == "summary" and not b.is_heading
        ]
        summary_text = "\n".join([b.text for b in summary_blocks])

        # Dynamic professional profile detection
        experience_roles = [
            e.role
            for e in experience_episodes
            if e.role
        ]

        full_resume_text = "\n".join(
            block.text for block in section_tagged_blocks
        )

        current_title, domain_category = self.resume_profile_extractor.extract(
            summary=summary_text,
            experience_roles=experience_roles,
            skills=extracted_skills,
            full_text=full_resume_text,
        )
        

        # Helper to extract clean line items for additional sections
        def extract_section_items(section_name: str) -> List[str]:
            blocks = [b for b in section_tagged_blocks if b.section == section_name and not b.is_heading]
            items = []
            for b in blocks:
                for l in b.text.split("\n"):
                    cleaned = self.cleaner.strip_bullet(l)
                    if not cleaned:
                        continue
                    if ";" in cleaned and len(cleaned.split(";")) >= 2:
                        parts = [p.strip().rstrip(". ") for p in cleaned.split(";") if p.strip()]
                        items.extend(parts)
                    else:
                        items.append(cleaned.rstrip(". "))
            return items

        # 6. Canonical Resume Construction
        resume = self.resume_builder.build(
            contact=contact_info,
            current_title=current_title,
            domain_category=domain_category,
            summary=summary_text,
            skills=extracted_skills,
            experience=experience_episodes,
            education=education_entries,
            projects=project_entries,
            certifications=cert_entries,
            publications=extract_section_items("publications"),
            awards=extract_section_items("awards"),
            achievements=extract_section_items("achievements"),
            languages=extract_section_items("languages"),
            interests=extract_section_items("interests"),
            volunteer_experience=extract_section_items("volunteer_experience"),
            references=extract_section_items("references"),
            metadata={
                "source_file": os.path.basename(file_path),
                "total_blocks_processed": len(section_tagged_blocks),
                "sections_detected": list(
                    set(
                        b.section
                        for b in section_tagged_blocks
                        if b.section
                    )
                ),
            },
        )

        # 7. Confidence & Evidence Assessment
        confidence_report = self.confidence_scorer.score_overall_resume(resume)

        # 8. Relational Episode Graph
        episode_graph = self.graph_builder.build_graph_from_episodes(experience_episodes)

        # 9. ATS Technical Analysis
        ats_result = self.ats_analyzer.analyze(resume, target_role_id=target_role)

        # 10. Optional Job Description Matching
        jd_match_result = None
        if jd_text and len(jd_text.strip()) > 10:
            jd_match_result = self.jd_matcher.match(resume, jd_text)
            ats_result.jd_similarity_score = jd_match_result.get("similarity_score")
            ats_result.jd_matched_keywords = jd_match_result.get("matched_skills", [])
            ats_result.jd_missing_keywords = jd_match_result.get("missing_skills", [])

        # Build clean JSON serializable output
        output = {
            "resume": resume.model_dump(),
            "ats_analysis": ats_result.model_dump(),
            "confidence": confidence_report,
            "episode_graph": episode_graph,
            "jd_match": jd_match_result,
            "layout_summary": {
                "total_blocks": len(section_tagged_blocks),
                "columns_detected": max([b.column_id for b in section_tagged_blocks], default=0) + 1,
                "sections_found": ats_result.sections_found
            }
        }

        return output

def analyze_resume(
    file_path: str,
    target_role: Optional[str] = None,
    jd_text: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience functional wrapper."""
    pipeline = ResumeAnalysisPipeline()
    return pipeline.analyze_resume(file_path, target_role=target_role, jd_text=jd_text)
