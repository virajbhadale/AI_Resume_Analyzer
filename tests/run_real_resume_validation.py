from pathlib import Path
from src.pipeline import ResumeAnalysisPipeline


SAMPLE_DIR = Path("data/samples")


def main():
    pipeline = ResumeAnalysisPipeline()

    resumes = sorted(
        [
            p
            for p in SAMPLE_DIR.iterdir()
            if p.suffix.lower() in {".pdf", ".docx"}
        ]
    )

    print("=" * 90)
    print("AI RESUME ANALYSER V2 — REAL RESUME VALIDATION")
    print("=" * 90)

    total = len(resumes)
    successful = 0
    failed = 0

    for resume_path in resumes:
        print("\n" + "=" * 90)
        print(f"FILE: {resume_path.name}")
        print("=" * 90)

        try:
            result = pipeline.analyze_resume(
                str(resume_path)
            )

            resume = result["resume"]

            experiences = resume.get(
                "experience",
                []
            )

            print(f"Status:          ✅ SUCCESS")
            print(f"Experiences:     {len(experiences)}")
            print(
                f"Skills:          "
                f"{len(resume.get('skills', []))}"
            )
            print(
                f"Education:       "
                f"{len(resume.get('education', []))}"
            )
            print(
                f"Projects:        "
                f"{len(resume.get('projects', []))}"
            )
            print(
                f"Certifications:  "
                f"{len(resume.get('certifications', []))}"
            )

            print("\nEXPERIENCES:")

            if not experiences:
                print("  NONE")

            for i, exp in enumerate(
                experiences,
                start=1
            ):
                print(f"\n  EXPERIENCE {i}")
                print(
                    f"    Role:       "
                    f"{exp.get('role')}"
                )
                print(
                    f"    Company:    "
                    f"{exp.get('company')}"
                )
                print(
                    f"    Location:   "
                    f"{exp.get('location')}"
                )
                print(
                    f"    Start:      "
                    f"{exp.get('start_date')}"
                )
                print(
                    f"    End:        "
                    f"{exp.get('end_date')}"
                )
                print(
                    f"    Current:    "
                    f"{exp.get('is_current')}"
                )
                print(
                    f"    Bullets:    "
                    f"{len(exp.get('bullet_points', []))}"
                )
                print(
                    f"    Confidence: "
                    f"{exp.get('confidence')}"
                )

            print("\nGRAPH:")

            graph = result.get(
                "episode_graph"
            )

            if isinstance(graph, dict):
                print(
                    f"  Nodes: "
                    f"{len(graph.get('nodes', []))}"
                )
                print(
                    f"  Edges: "
                    f"{len(graph.get('edges', []))}"
                )
            else:
                print(f"  {type(graph).__name__}")

            ats = result.get(
                "ats_analysis",
                {}
            )

            print("\nATS:")

            if isinstance(ats, dict):
                print(
                    f"  Overall Score: "
                    f"{ats.get('overall_score')}"
                )
                print(
                    f"  Sections Found: "
                    f"{ats.get('sections_found')}"
                )

            successful += 1

        except Exception as exc:
            failed += 1

            print("Status:          ❌ FAILED")
            print(
                f"Exception:       "
                f"{type(exc).__name__}"
            )
            print(
                f"Message:         "
                f"{exc}"
            )

    print("\n" + "=" * 90)
    print("FINAL REAL-RESUME VALIDATION SUMMARY")
    print("=" * 90)

    print(f"Total resumes:   {total}")
    print(f"Successful:      {successful}")
    print(f"Failed:          {failed}")

    if failed == 0:
        print(
            "\n🎉 ALL REAL RESUMES PROCESSED SUCCESSFULLY"
        )
    else:
        print(
            "\n⚠️ SOME REAL RESUMES FAILED"
        )


if __name__ == "__main__":
    main()