# Project Viva & Interview Defense Guide — AI Resume Analyser V2

This document provides concise, technically rigorous explanations and answers for academic vivas, project presentations, and technical job interviews.

---

## 1. 30-Second Elevator Pitch

> "AI Resume Analyser V2 is a layout-aware, context-aware resume intelligence system tailored for Data Science and Technical engineering roles. Unlike naive parsers that merge text into disordered blobs, our system extracts geometric bounding boxes and font metadata to resolve complex multi-column layouts, uses an Anchor Triplet graph `[ROLE + COMPANY + DATE]` to prevent distinct work experiences from merging, and performs evidence-based ATS competency analysis across 8 technical job taxonomies without relying on fragile external APIs or black-box scores."

---

## 2. 1-Minute Technical Summary

> "The core challenge in automated resume parsing is structural ambiguity: multi-column formats interleave unrelated sections, and sequential bullet points easily cause multiple jobs to collapse into one. 
> 
> To solve this, our pipeline operates in 6 deterministic and semantic stages:
> 1. **Geometric Ingestion:** PyMuPDF and python-docx capture spatial coordinates `(x0, y0, x1, y1)` and font metrics.
> 2. **Layout Reconstruction:** An X-histogram clustering engine detects column gutters and orders blocks into true human reading order.
> 3. **Section Tagging:** A RapidFuzz and typographical heading classifier tags blocks with canonical sections.
> 4. **Anti-Merge Experience Segmentation:** We identify Role and Company anchors bound to Date ranges. A new anchor candidate establishes an episode boundary, preventing accidental concatenation.
> 5. **Evidence Fusion:** Verifiable confidence scores are computed from typographical emphasis and pattern certainty.
> 6. **ATS & Role Alignment:** Resumes are matched against 8 curated data/software role profiles, generating actionable skill gap insights."

---

## 3. 3-Minute Comprehensive Viva Presentation

> "Good morning examiners. Today I am presenting **AI Resume Analyser V2**.
> 
> ### The Problem:
> Most existing resume parsers are essentially 'PDF-to-regex' scripts. In practice, they fail in two catastrophic ways:
> 1. **The Two-Column Failure:** In a standard two-column resume (with skills on the left and experience on the right), standard PDF extractors read horizontally across the page, interleaving skills with job descriptions.
> 2. **The Experience Merge Failure:** Multiple sequential jobs (such as an internship followed by a full-time role) get lumped together into a single job description.
> 
> ### Our Solution:
> We designed a multi-stage architecture based on Computer Science and Data Science principles:
> 
> - **Spatial Awareness:** We extract the physical bounding box for every token. By analyzing the horizontal distribution of block centroids, we calculate if a page is single-column or multi-column, and read columns in natural vertical sequence before traversing horizontally.
> - **Anchor-Directed Episode Segmentation:** We model employment history as an anchor-directed graph. An episode boundary requires a new Anchor Candidate: a validated Job Role or Organization name linked with a distinct Date range. Bullet points are bound strictly to their active anchor.
> - **Taxonomy-Driven Skill Extraction:** Instead of crude keyword matching, we maintain normalized skill taxonomies for 8 technical role families—including Data Science, MLOps, and Generative AI. We distinguish between core requirements (e.g., PyTorch, Transformers, SQL) and secondary competencies.
> - **Transparent ATS Scoring:** We do not generate arbitrary black-box percentages. Our ATS score is mathematically explainable, combining Section Completeness (30%), Skill Alignment (40%), Experience Depth (25%), and Formatting Integrity (5%).
> 
> ### Evaluation & Feasibility:
> - The entire system executes on standard CPU hardware in under 400 milliseconds.
> - It is 100% reproducible and tested via automated regression test suites verifying that $N$ separate jobs remain $N$ distinct episodes."

---

## 4. Architectural Deep Dive: Key Technical Decisions

### Why PyMuPDF over `pypdf` or `pdfminer`?
- `pypdf` only provides raw string dumps, losing coordinate and font metadata.
- `pdfminer` is significantly slower and memory-heavy.
- `PyMuPDF (fitz)` provides sub-millisecond C-level bounding box extraction `(x0, y0, x1, y1)`, font size, and typographical flags (bold/italic) necessary for heading and anchor detection.

### Why Anchor Triplet Graph over Unconstrained Clustering?
- Unconstrained ML clustering (such as K-Means or DBSCAN on text embeddings) frequently fails because bullet points of different jobs often share identical semantic embeddings (e.g. 'Built REST APIs in Python').
- Our Anchor Triplet `[ROLE + COMPANY + DATE]` treats structural delimiters as hard episodic boundaries, designed to preserve separate job episodes when distinct role/company/date anchors are present.

### Why not rely solely on an LLM API?
- **Cost & Latency:** An LLM call takes 2–8 seconds per resume and costs API credits on every run.
- **Privacy & GDPR:** Resumes contain PII (names, phone numbers, addresses) which should not be sent over external networks unnecessarily.
- **Reproducibility:** LLM outputs suffer from non-deterministic hallucinations (dropping bullet points or inventing job titles). Our deterministic pipeline is consistent, explainable, and instant.

---

## 5. Likely Viva Questions & Model Answers

### Q1: "How do you distinguish between a section heading and regular bold text?"
**Model Answer:**
> "We combine three signals:
> 1. **Word Count & Length Constraint:** Real section headings rarely exceed 1 to 4 words and under 30 characters.
> 2. **Fuzzy String Similarity:** We compare the normalized text against a canonical YAML synonym dictionary using RapidFuzz token sort ratio with an 82% threshold.
> 3. **Typographical Boost:** We add confidence bonuses (+0.08 for bold flags, +0.08 for all-uppercase, and +0.05 for font size $\ge 12\text{pt}$). A regular bold sentence within a paragraph fails the length and dictionary constraints."

### Q2: "How do you handle two-column resumes where one column is narrower than the other?"
**Model Answer:**
> "Our `LayoutEngine` evaluates block centroid X-coordinates against page width. Rather than assuming a rigid 50/50 split, it detects both symmetric columns and asymmetric sidebar layouts (e.g., 35% left sidebar, 65% main content) by calculating regional block density. Blocks within Column 0 are ordered by Y-coordinate before blocks in Column 1 are evaluated."

### Q3: "How does the ATS score work, and why is it not arbitrary?"
**Model Answer:**
> "Many online tools display a random 'ATS Score: 87%' with zero mathematical justification. In our system, the ATS score is a transparent weighted formula:
> $$\text{Score} = 0.30 \times S_{\text{sections}} + 0.40 \times S_{\text{skills}} + 0.25 \times S_{\text{experience}} + 0.05 \times S_{\text{contact}}$$
> Each component is directly inspectable in the UI. For instance, the skills component measures the exact percentage of core competencies satisfied for the selected role profile."

### Q4: "What happens if a resume is an image scan without digital text?"
**Model Answer:**
> "Our parser counts the total digital characters extracted across all pages. If the count is below 40 characters, the system raises a typed `ScannedDocumentError` and notifies the user that the document is an image scan rather than failing silently or returning garbage characters."
