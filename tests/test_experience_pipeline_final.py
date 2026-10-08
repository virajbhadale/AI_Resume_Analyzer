from src.document_models import DocumentBlock
from src.inference.segmenter import ExperienceSegmenter


def run_test(name, blocks, expected_count, expected_checks=None):
    print("\n" + "=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)

    segmenter = ExperienceSegmenter()
    if name.startswith("4"):
        print("\nDEBUG TEST 4 INPUT:")
        for b in blocks:
            print(
                b.id,
                repr(b.text),
                "bold=",
                b.is_bold,
                "section=",
                b.section,
            )
    results = segmenter.segment_experiences(blocks)

    passed = True

    # ---------------------------------------------------------
    # COUNT CHECK
    # ---------------------------------------------------------

    if len(results) != expected_count:
        passed = False
        print(
            f"❌ COUNT FAILED | Expected {expected_count}, "
            f"Got {len(results)}"
        )
    else:
        print(f"✅ COUNT PASSED | {len(results)} experience(s)")

    # ---------------------------------------------------------
    # ADDITIONAL CHECKS
    # ---------------------------------------------------------

    if expected_checks:
        for description, check in expected_checks:

            try:
                result = check(results)

                if result:
                    print(f"✅ {description}")
                else:
                    print(f"❌ {description}")
                    passed = False

            except Exception as e:
                print(f"❌ {description} | ERROR: {e}")
                passed = False

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    for i, experience in enumerate(results, 1):
        print(f"\nEXPERIENCE {i}")
        print(f"  Role:     {experience.role}")
        print(f"  Company:  {experience.company}")
        print(f"  Location: {experience.location}")
        print(
            f"  Dates:    "
            f"{experience.start_date} → {experience.end_date}"
        )
        print(f"  Current:  {experience.is_current}")
        print(f"  Bullets:  {len(experience.bullet_points)}")

    print("\nRESULT:", "✅ PASS" if passed else "❌ FAIL")

    return passed


# =============================================================
# TEST 1 — MIXED EXPERIENCE FORMATS
# =============================================================

test1_blocks = [

    # Normal role/company/date format
    DocumentBlock(
        id="t1_b1",
        text="Data Analyst Intern",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t1_b2",
        text="ComplyHappy",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t1_b3",
        text="Jun 2023 – Aug 2023",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t1_b4",
        text="Analyzed business data using Python and SQL.",
        is_bold=False,
        section="experience",
    ),

    # Role @ company
    DocumentBlock(
        id="t1_b5",
        text="Software Engineer @ TechNova Solutions",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t1_b6",
        text="Jan 2024 – Dec 2024",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t1_b7",
        text="Developed backend APIs using Python and FastAPI.",
        is_bold=False,
        section="experience",
    ),

    # Role-company-date compact
    DocumentBlock(
        id="t1_b8",
        text=(
            "AI Engineer - FutureTech Labs "
            "(Jan 2025 – Present)"
        ),
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t1_b9",
        text="Built production machine learning systems.",
        is_bold=False,
        section="experience",
    ),
]


test1 = run_test(
    "1 — Mixed Experience Formats",
    test1_blocks,
    expected_count=3,
)


# =============================================================
# TEST 2 — SAME ROLE, DIFFERENT COMPANIES
# =============================================================

test2_blocks = [

    DocumentBlock(
        id="t2_b1",
        text="Data Analyst Intern",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t2_b2",
        text="Company Alpha",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t2_b3",
        text="Jan 2023 – Jun 2023",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t2_b4",
        text="Analyzed datasets using Python and SQL.",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t2_b5",
        text="Data Analyst Intern",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t2_b6",
        text="Company Beta",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t2_b7",
        text="Jul 2023 – Dec 2023",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t2_b8",
        text="Created dashboards using Power BI and Excel.",
        is_bold=False,
        section="experience",
    ),
]


test2 = run_test(
    "2 — Same Role, Different Companies",
    test2_blocks,
    expected_count=2,
    expected_checks=[
        (
            "Companies remain separate",
            lambda r: (
                r[0].company == "Company Alpha"
                and r[1].company == "Company Beta"
            ),
        ),
    ],
)


# =============================================================
# TEST 3 — COMPLETE EVIDENCE FUSION
# ROLE + COMPANY + LOCATION + DATE + MANY BULLETS
# =============================================================

test3_blocks = [

    DocumentBlock(
        id="t3_b1",
        text="Machine Learning Engineer",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t3_b2",
        text="AI Systems Pvt. Ltd. (Location - Bengaluru, India)",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t3_b3",
        text="Jan 2024 – Present",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t3_b4",
        text="Built machine learning pipelines using Python.",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t3_b5",
        text="Developed APIs using FastAPI.",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t3_b6",
        text="Deployed models using Docker.",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t3_b7",
        text="Implemented monitoring for production models.",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t3_b8",
        text="Improved inference performance.",
        is_bold=False,
        section="experience",
    ),
]


test3 = run_test(
    "3 — Complete Evidence Fusion",
    test3_blocks,
    expected_count=1,
    expected_checks=[
        (
            "Role detected correctly",
            lambda r: r[0].role == "Machine Learning Engineer",
        ),
        (
            "Company detected correctly",
            lambda r: r[0].company == "AI Systems Pvt. Ltd.",
        ),
        (
            "Location detected correctly",
            lambda r: r[0].location == "Bengaluru, India",
        ),
        (
            "Date detected correctly",
            lambda r: (
                r[0].start_date == "Jan 2024"
                and r[0].end_date == "Present"
            ),
        ),
        (
            "All 5 bullets preserved",
            lambda r: len(r[0].bullet_points) == 5,
        ),
    ],
)


# =============================================================
# TEST 4 — MESSY / REAL-WORLD FORMATTING
# =============================================================

test4_blocks = [

    DocumentBlock(
        id="t4_b1",
        text="  Senior   Data Engineer  ",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t4_b2",
        text="TechNova Solutions",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t4_b3",
        text="(Jan 2024 â€“ Present)",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t4_b4",
        text="• Built scalable ETL pipelines using Python.",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t4_b5",
        text="– Automated data processing workflows.",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t4_b6",
        text="  Optimized database queries and performance.  ",
        is_bold=False,
        section="experience",
    ),
]


test4 = run_test(
    "4 — Messy / Real-World Formatting",
    test4_blocks,
    expected_count=1,
    expected_checks=[
        (
            "Role detected",
            lambda r: r[0].role == "Senior Data Engineer",
        ),
        (
            "Company detected",
            lambda r: r[0].company == "TechNova Solutions",
        ),
        (
            "Description preserved",
            lambda r: len(r[0].bullet_points) == 3,
        ),
    ],
)


# =============================================================
# TEST 5 — FALSE POSITIVE / NON-EXPERIENCE SECTIONS
# =============================================================

test5_blocks = [

    # Projects
    DocumentBlock(
        id="t5_b1",
        text="AI Engineer",
        is_bold=True,
        section="projects",
    ),

    DocumentBlock(
        id="t5_b2",
        text="Built an AI chatbot using Python.",
        is_bold=False,
        section="projects",
    ),

    # Education
    DocumentBlock(
        id="t5_b3",
        text="Research Assistant",
        is_bold=True,
        section="education",
    ),

    DocumentBlock(
        id="t5_b4",
        text="MIT Research Lab",
        is_bold=True,
        section="education",
    ),

    # Certifications
    DocumentBlock(
        id="t5_b5",
        text="AWS Solutions Architect",
        is_bold=True,
        section="certifications",
    ),

    # Awards
    DocumentBlock(
        id="t5_b6",
        text="Best Software Engineer Award",
        is_bold=True,
        section="awards",
    ),
]


test5 = run_test(
    "5 — False Positive / Non-Experience Sections",
    test5_blocks,
    expected_count=0,
)


# =============================================================
# TEST 6 — ORIGINAL REAL-WORLD REGRESSION
# COMPLYHAPPY → BYZWIZ → MARVA AI
# =============================================================

test6_blocks = [

    # ---------------------------------------------------------
    # EXPERIENCE 1
    # ---------------------------------------------------------

    DocumentBlock(
        id="t6_b1",
        text="Data Analyst Intern",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b2",
        text="ComplyHappy",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b3",
        text="19 Jun 2023 – 11 Aug 2023",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b4",
        text=(
            "Analyzed business data and created reports "
            "using Python, SQL and Excel."
        ),
        is_bold=False,
        section="experience",
    ),

    # ---------------------------------------------------------
    # EXPERIENCE 2
    # ---------------------------------------------------------

    DocumentBlock(
        id="t6_b5",
        text="Data Analyst Intern",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b6",
        text="ByzWiz",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b7",
        text="12 Jun 2024 – 07 Aug 2024",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b8",
        text=(
            "Performed data analysis and developed "
            "dashboards to support business decisions."
        ),
        is_bold=False,
        section="experience",
    ),

    # ---------------------------------------------------------
    # EXPERIENCE 3
    # ---------------------------------------------------------

    DocumentBlock(
        id="t6_b9",
        text="AI Engineer Intern",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b10",
        text="Marva AI (Location - Pune, India)",
        is_bold=True,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b11",
        text="06 Jan 2025 – 17 May 2025",
        is_bold=False,
        section="experience",
    ),

    DocumentBlock(
        id="t6_b12",
        text=(
            "Developed and deployed machine learning "
            "models using Python and FastAPI."
        ),
        is_bold=False,
        section="experience",
    ),
]


test6 = run_test(
    "6 — Original Regression: ComplyHappy → ByzWiz → Marva AI",
    test6_blocks,
    expected_count=3,
    expected_checks=[
        (
            "Experience 1 company = ComplyHappy",
            lambda r: r[0].company == "ComplyHappy",
        ),
        (
            "Experience 2 company = ByzWiz",
            lambda r: r[1].company == "ByzWiz",
        ),
        (
            "Experience 3 company = Marva AI",
            lambda r: r[2].company == "Marva AI",
        ),
        (
            "Experience 1 role = Data Analyst Intern",
            lambda r: r[0].role == "Data Analyst Intern",
        ),
        (
            "Experience 2 role = Data Analyst Intern",
            lambda r: r[1].role == "Data Analyst Intern",
        ),
        (
            "Experience 3 role = AI Engineer Intern",
            lambda r: r[2].role == "AI Engineer Intern",
        ),
        (
            "Marva AI location preserved",
            lambda r: r[2].location == "Pune, India",
        ),
        (
            "All three have dates",
            lambda r: all(x.start_date for x in r),
        ),
    ],
)


# =============================================================
# FINAL SUMMARY
# =============================================================

all_tests = [
    test1,
    test2,
    test3,
    test4,
    test5,
    test6,
]

passed = sum(all_tests)
total = len(all_tests)

print("\n")
print("=" * 70)
print("FINAL EXPERIENCE PIPELINE REGRESSION SUMMARY")
print("=" * 70)

print(f"Passed: {passed}/{total}")
print(f"Failed: {total - passed}/{total}")

if passed == total:
    print("\n🎉 ALL 6 TESTS PASSED")
    print("Experience segmentation regression suite: PASS")
else:
    print("\n⚠️ SOME TESTS FAILED")
    print("Do NOT declare the Experience Pipeline ready yet.")

print("=" * 70)