from src.pipeline import ResumeAnalysisPipeline
import os
import sys

FILES = [
    r"data\samples\Bhavesh_Wadhwani_Resume.pdf",
    r"data\samples\Jillani Resume.pdf",
    r"data\samples\Master_Sample_Resume.pdf",
    r"data\samples\syzfjbzwjncs.pdf",
    r"data\samples\Viraj_Resume.pdf",
    r"data\samples\yrhtcnjyzgsf.pdf",
]

def main():
    pipeline = ResumeAnalysisPipeline()

    print("=" * 90)
    print("AI RESUME ANALYSER V2 - SELECTED REAL RESUME VALIDATION")
    print("=" * 90)

    successful = 0
    failed = 0

    for file_path in FILES:

        print("\n" + "=" * 90)
        print(f"FILE: {os.path.basename(file_path)}")
        print("=" * 90)

        if not os.path.exists(file_path):
            print(f"FILE NOT FOUND: {file_path}")
            failed += 1
            continue

        try:
            result = pipeline.analyze_resume(file_path)

            resume = result["resume"]
            experiences = resume.get("experience", [])

            print(f"Experiences:     {len(experiences)}")

            print("\nEXPERIENCES:")

            for i, exp in enumerate(experiences, 1):
                print(f"\n  EXPERIENCE {i}")
                print(f"    Role:       {exp.get('role')}")
                print(f"    Company:    {exp.get('company')}")
                print(f"    Location:   {exp.get('location')}")
                print(f"    Start:      {exp.get('start_date')}")
                print(f"    End:        {exp.get('end_date')}")
                print(f"    Current:    {exp.get('is_current')}")
                print(f"    Bullets:    {len(exp.get('bullet_points', []))}")
                print(f"    Confidence: {exp.get('confidence')}")

            ats = result.get("ats_analysis", {})

            print("\nATS:")
            print(f"  Overall Score: {ats.get('overall_score')}")
            print(f"  Sections Found: {ats.get('sections_found')}")

            graph = result.get("episode_graph", {})

            if isinstance(graph, dict):
                print("\nGRAPH:")
                print(f"  Nodes: {len(graph.get('nodes', []))}")
                print(f"  Edges: {len(graph.get('edges', []))}")

            successful += 1
            print("\nSTATUS: SUCCESS")

        except Exception as e:
            failed += 1
            print("\nSTATUS: FAILED")
            print(f"ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 90)
    print("FINAL SELECTED REAL-RESUME VALIDATION SUMMARY")
    print("=" * 90)

    print(f"Total resumes:   {len(FILES)}")
    print(f"Successful:      {successful}")
    print(f"Failed:          {failed}")

    if failed == 0:
        print("\nALL SELECTED REAL RESUMES PROCESSED SUCCESSFULLY")
    else:
        print("\nSOME REAL RESUMES FAILED")

if __name__ == "__main__":
    main()
