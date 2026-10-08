#!/usr/bin/env python3
"""CLI entry point for AI Resume Analyser V2."""
import os
import sys
import json
import argparse
from src.pipeline import analyze_resume

def main():
    parser = argparse.ArgumentParser(
        description="AI Resume Analyser V2 — Layout-aware Technical Resume Intelligence System"
    )
    parser.add_argument("resume_path", help="Path to input resume file (.pdf, .docx, .txt)")
    parser.add_argument(
        "--role",
        default=None,
        choices=[
            "data_scientist", "ml_engineer", "ai_engineer", "genai_engineer",
            "data_engineer", "data_analyst", "mlops_engineer", "software_engineer"
        ],
        help="Target Technical Role profile (default: data_scientist)"
    )
    parser.add_argument("--jd", default=None, help="Optional text or path to Job Description file")
    parser.add_argument("--output", default=None, help="Output path for structured JSON (default: stdout/outputs/)")

    args = parser.parse_args()

    if not os.path.exists(args.resume_path):
        print(f"[ERROR] Resume file not found: {args.resume_path}", file=sys.stderr)
        sys.exit(1)

    jd_content = None
    if args.jd:
        if os.path.exists(args.jd):
            with open(args.jd, "r", encoding="utf-8") as f:
                jd_content = f.read()
        else:
            jd_content = args.jd

    print(f"\n=======================================================")
    print(f"            AI RESUME ANALYSER V2")
    print(f"=======================================================")
    print(f"[*] Ingesting: {args.resume_path}")
    print(
        f"[*] Target Role: "
        f"{args.role if args.role else 'Auto-detect from resume'}"
    )

    try:
        results = analyze_resume(args.resume_path, target_role=args.role, jd_text=jd_content)
    except Exception as e:
        print(f"[FATAL] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    resume = results["resume"]
    ats = results["ats_analysis"]
    conf = results["confidence"]

    print("\n-------------------------------------------------------")
    print("                 EXTRACTION SUMMARY")
    print("-------------------------------------------------------")
    print(f"Candidate Name:    {resume['contact']['name'] or 'Not Detected'}")
    print(f"Email:             {resume['contact']['email'] or 'Not Detected'}")
    print(f"Phone:             {resume['contact']['phone'] or 'Not Detected'}")
    print(f"LinkedIn:          {resume['contact']['linkedin'] or 'Not Detected'}")
    print(f"Work Experiences:  {len(resume['experience'])} episode(s)")
    print(f"Education Count:   {len(resume['education'])} degree(s)")
    print(f"Skills Detected:   {len(resume['skills'])} skill(s)")
    print(f"Overall Confidence:{conf['overall_confidence'] * 100:.0f}%")

    print("\n-------------------------------------------------------")
    print(f"           ATS ANALYSIS FOR: {ats['target_role'].upper()}")
    print("-------------------------------------------------------")
    print(f"Overall ATS Score:         {ats['overall_score']}/100")
    print(f"Section Completeness:      {ats['section_completeness_score']}%")
    print(f"Core Skills Match:         {ats['skill_coverage_score']}%")
    print(f"Matched Core Skills:       {', '.join(ats['detected_core_skills']) if ats['detected_core_skills'] else 'None'}")
    print(f"Missing Core Competencies: {', '.join(ats['missing_core_skills']) if ats['missing_core_skills'] else 'None'}")

    if ats['parsing_risks']:
        print("\n[!] Parsing Risk Warnings:")
        for r in ats['parsing_risks']:
            print(f"    - [{r['level']}] {r['category']}: {r['message']}")

    # Save output
    out_dir = "data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, f"structured_{os.path.basename(args.resume_path)}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Full structured JSON saved to: {out_path}\n")

if __name__ == "__main__":
    main()
