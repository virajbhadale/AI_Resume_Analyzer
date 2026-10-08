# AI Resume Analyser V2 — Technical Resume Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-CPU%20%7C%20Windows%20%7C%20Colab-green.svg)](https://colab.research.google.com/)
[![Tests](https://img.shields.io/badge/tests-15%20passed-emerald.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview

**AI Resume Analyser V2** is an academic- and portfolio-grade, layout-aware resume intelligence system engineered specifically for **Data Science, Machine Learning, AI Engineering, Generative AI, Data Engineering, Data Analytics, MLOps, and Python Technical Software Engineering** roles.

Traditional resume parsers often fail on two-column layouts and can merge distinct employment histories into a single disorganized paragraph. AI Resume Analyser V2 addresses these failure modes through a **geometric layout reconstruction engine**, an **Anchor Triplet `[ROLE + COMPANY + DATE]` segmentation graph**, configurable section detection, normalized technical-skill taxonomies, and explainable ATS evaluation. The pipeline is deterministic for digital documents; OCR is an optional fallback for scanned/image resumes and can use the configured Gemini OCR service.

---

## 2. Core Architecture & Pipeline Flow

```
[Raw Document: PDF / DOCX / TXT]
            │
            ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Document Ingestion & Geometry       │  (PyMuPDF & python-docx)
│   • Extracts bounding boxes (x0, y0, x1, y1) │
│   • Captures font metrics (size, weight)     │
│   • Detects scanned/empty documents          │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Layout Reconstruction & Hygiene     │  (X-Histogram Bimodal Clustering)
│   • Detects single-column vs. two-column     │
│   • Reorders blocks into true reading order  │
│   • Strips bullets, artifacts & page numbers │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 3: Section Boundary Tagging            │  (RapidFuzz + Typographical Bonus)
│   • Config-driven canonical section registry │
│   • Fuzzy heading classification (cutoff: 82)│
│   • Contextual section propagation           │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 4: Anti-Merge Experience Segmentation  │  (Anchor Triplet Graph Engine)
│   • Anchor Triplet: [ROLE, COMPANY, DATE]    │
│   • Bounded episode isolation                │
│   • Strictly isolates N distinct jobs        │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 5: Entity Fusion & Confidence          │  (Evidence Fusion)
│   • Contact, Skills, Education, Projects     │
│   • Verifiable confidence calculations       │
│   • CanonicalResume Pydantic Schema          │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 6: Domain ATS & Role Analysis          │  (8 Technical Role Profiles)
│   • Core competencies coverage check         │
│   • Missing technical skills identification  │
│   • Optional Job Description (JD) matching   │
└──────────────────────────────────────────────┘
```

---

## 3. Key Capabilities & Technical Innovations

1. **Gutter & Two-Column Awareness:** Solves the classic two-column reading trap by computing bimodal X-center distributions. Left-column blocks are sequentially read prior to right-column blocks.
2. **Anchor-Directed Episode Segmentation:** Prevents accidental merging of distinct internships and corporate roles. Every job is bound to a validated Anchor Candidate.
3. **8 Specialized Technical Role Taxonomies:**
   - Data Scientist
   - Machine Learning Engineer
   - AI Engineer
   - Generative AI Engineer (RAG, LangChain, Vector DBs)
   - Data Engineer (Spark, Airflow, ETL)
   - Data Analyst (Power BI, Tableau, SQL)
   - MLOps Engineer (Docker, Kubernetes, MLflow, Drift)
   - Software Engineer (Python, FastAPI, System Design)
4. **Verifiable ATS Scoring (No Fabricated Metrics):**
   - Section Completeness (30%)
   - Technical Skill Coverage (40%)
   - Experience Depth (25%)
   - Contact / Formatting Integrity (5%)
5. **Local-first processing with optional OCR:** Digital PDF/DOCX/TXT parsing and ATS analysis run locally on CPU. Scanned/image OCR uses the configured Gemini OCR integration when enabled; the application does not require a GPU for the core pipeline.

---

## 4. Project Structure

```
AI_Resume_Analyser_V2/
│
├── config/
│   ├── sections.yaml              # Synonyms & canonical section registry
│   ├── job_roles.yaml             # 8 Technical role skill taxonomies
│   └── settings.yaml              # Pipeline weights, layout thresholds
│
├── src/
│   ├── document_models.py         # Pydantic v2 schemas: BBox, DocumentBlock, CanonicalResume
│   ├── cleaner.py                 # Hygiene engine (strips page nums, bullet chars)
│   ├── layout.py                  # Column detector & 2D reading-order sorter
│   ├── parser.py                  # PyMuPDF & python-docx unified ingestor
│   ├── section_extractor.py       # Section heading classifier (RapidFuzz + typography)
│   │
│   ├── extraction/
│   │   ├── contact.py             # Name, email, phone, LinkedIn, GitHub extractor
│   │   ├── dates.py               # Multi-format date parser & range normalizer
│   │   ├── education.py           # Degrees, universities, GPAs
│   │   ├── skills.py              # Taxonomy-based technical skill matcher
│   │   ├── projects.py            # Technical projects & repo link extractor
│   │   ├── certifications.py      # Certifications & issuing bodies
│   │   └── experience.py          # Orchestrates episodic experience construction
│   │
│   ├── inference/
│   │   ├── anchors.py             # Role/Company/Date candidate detection
│   │   ├── segmenter.py           # Anti-merge episode boundary engine
│   │   ├── confidence.py          # Verifiable confidence scoring
│   │   └── graph.py               # NetworkX relational episode graph
│   │
│   ├── ats/
│   │   ├── analyzer.py            # ATS completeness & skill gap analyzer
│   │   └── jd_matcher.py          # Job description TF-IDF & keyword matcher
│   │
│   └── pipeline.py                # Main programmatic API: analyze_resume()
│
├── tests/                         # Pytest automated test suite (11/11 passing)
│   ├── test_experience_segmentation.py  # Anti-merging regression tests
│   ├── test_sections.py
│   ├── test_layout.py
│   ├── test_skills.py
│   ├── test_ats.py
│   └── test_parser.py
│
├── data/
│   ├── samples/                   # Synthetic test resumes (Data Scientist, MLOps, GenAI)
│   └── outputs/                   # Exported structured JSON resumes
│
├── notebooks/
│   └── development.ipynb          # Step-by-step Google Colab notebook
│
├── run.py                         # Standalone CLI entry point
├── server.ts                      # Express API backend & dev server
├── requirements.txt               # Pinned Python dependencies
├── README.md                      # Comprehensive documentation
└── PROJECT_EXPLANATION.md         # Viva defense and interview preparation guide
```

---

## 5. Installation & Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+ (for Web UI)

### 1. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/AI_Resume_Analyser_V2.git
cd AI_Resume_Analyser_V2

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Pytest Suite
```bash
pytest tests/ -v
```
*All 11 unit & regression tests will execute and pass in $< 0.5$ seconds.*

---

## 6. CLI Usage

You can analyze any resume from your terminal using `run.py`:

```bash
# Analyze a resume for Data Scientist role
python run.py data/samples/sample_resume_1.txt --role data_scientist

# Analyze for MLOps Engineer
python run.py data/samples/sample_resume_2.txt --role mlops_engineer

# Analyze for Generative AI Engineer with Job Description
python run.py data/samples/sample_resume_3.txt --role genai_engineer --jd "Must know LangChain, Pinecone, and RAG"
```

Output:
```text
=======================================================
            AI RESUME ANALYSER V2
=======================================================
[*] Ingesting: data/samples/sample_resume_1.txt
[*] Target Role: data_scientist
-------------------------------------------------------
                 EXTRACTION SUMMARY
-------------------------------------------------------
Candidate Name:    ALEXANDER CHEN
Email:             alex.chen@email.com
Phone:             +1 (555) 234-5678
LinkedIn:          linkedin.com/in/alexchen-ds
Work Experiences:  3 episode(s)
Education Count:   2 degree(s)
Skills Detected:   24 skill(s)
Overall Confidence:92%
-------------------------------------------------------
           ATS ANALYSIS FOR: DATA SCIENTIST
-------------------------------------------------------
Overall ATS Score:         92.0/100
Section Completeness:      100.0%
Core Skills Match:         80.0%
Matched Core Skills:       Python, SQL, Pandas, NumPy, Scikit-Learn, Statistics, Machine Learning, Exploratory Data Analysis, Feature Engineering
Missing Core Competencies: Data Visualization
[+] Full structured JSON saved to: data/outputs/structured_sample_resume_1.txt.json
```

---

## 7. Google Colab Execution

To run in Google Colab:
1. Open Google Colab (`https://colab.research.google.com`).
2. Upload `notebooks/development.ipynb` and the project directory.
3. Run the top install cell:
   ```python
   !pip install pymupdf python-docx pydantic rapidfuzz networkx scikit-learn pyyaml pytest
   ```
4. Follow the interactive step-by-step cells for parsing, anti-merge validation, and ATS breakdown.

---

## 8. Web Dashboard

Launch the interactive web application:
```bash
npm install
npm run dev
```
Open `http://localhost:3000` to inspect:
- Interactive role selection & benchmark resumes
- Visual work experience timeline (unmerged episodes)
- Core skill alignment pills (Matched vs Missing)
- Relational NetworkX episode graph
- Full Pydantic v2 structured JSON viewer and export

---

## 9. License

MIT License. Developed for academic demonstration, technical viva evaluation, and portfolio showcase.
