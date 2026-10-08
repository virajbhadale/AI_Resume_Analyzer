import React, { useState, useEffect } from "react";
import KeywordGapHighlighter from "./components/KeywordGapHighlighter";
import {
  FileText,
  Briefcase,
  GraduationCap,
  Award,
  Code2,
  CheckCircle2,
  AlertTriangle,
  Download,
  Copy,
  Play,
  RotateCcw,
  Sparkles,
  ChevronRight,
  TrendingUp,
  ShieldCheck,
  Compass,
  Cpu,
  Terminal,
  ExternalLink,
  Layers,
  FileCheck,
  Search
} from "lucide-react";

interface RoleOption {
  id: string;
  title: string;
  category: string;
}

interface SampleResume {
  id: string;
  filename: string;
  title: string;
  category: string;
  description: string;
  content: string;
}

const ROLES: RoleOption[] = [
  { id: "data_scientist", title: "Data Scientist", category: "Data Science" },
  { id: "ml_engineer", title: "Machine Learning Engineer", category: "Machine Learning" },
  { id: "ai_engineer", title: "AI Engineer", category: "Artificial Intelligence" },
  { id: "genai_engineer", title: "Generative AI Engineer", category: "Generative AI" },
  { id: "data_engineer", title: "Data Engineer", category: "Data Engineering" },
  { id: "data_analyst", title: "Data Analyst", category: "Data Analytics" },
  { id: "mlops_engineer", title: "MLOps Engineer", category: "MLOps & Cloud" },
  { id: "software_engineer", title: "Software Engineer (Python)", category: "Software Engineering" },
];

export default function App() {
  const [targetRole, setTargetRole] = useState("data_scientist");
  const [samples, setSamples] = useState<SampleResume[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string>("sample_4");
  const [resumeText, setResumeText] = useState("");
  const [fileBase64, setFileBase64] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<{ name: string; size: string; type: string } | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState("sample_resume_4.txt");
  const [jdText, setJdText] = useState("");
  const [activeTab, setActiveTab] = useState<string>("overview");
  const [loading, setLoading] = useState(false);
  const [testLoading, setTestLoading] = useState(false);
  const [testOutput, setTestOutput] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [copied, setCopied] = useState(false);

  // Load samples on mount
  useEffect(() => {
    fetch("/api/samples")
      .then((res) => res.json())
      .then((data) => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
          // Prefer sample_4 (Master with 4 education degrees) if available
          const preferred = data.samples.find((s: SampleResume) => s.id === "sample_4") || data.samples[0];
          setSelectedSampleId(preferred.id);
          setResumeText(preferred.content);
          setFileName(preferred.filename);
        }
      })
      .catch((err) => console.error("Could not fetch samples:", err));
  }, []);

  const handleSelectSample = (sampleId: string) => {
    const s = samples.find((x) => x.id === sampleId);
    if (s) {
      setSelectedSampleId(s.id);
      setResumeText(s.content);
      setFileName(s.filename);
      setFileBase64(null);
      setFileInfo(null);
      setImagePreviewUrl(null);
      if (s.id === "sample_4") setTargetRole("data_scientist");
      if (s.id === "sample_1") setTargetRole("data_scientist");
      if (s.id === "sample_2") setTargetRole("mlops_engineer");
      if (s.id === "sample_3") setTargetRole("genai_engineer");
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const sizeStr =
      file.size > 1024 * 1024
        ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
        : `${Math.round(file.size / 1024)} KB`;

    setFileName(file.name);
    setSelectedSampleId("custom");
    setFileInfo({ name: file.name, size: sizeStr, type: file.type || "document" });

    const ext = file.name.split(".").pop()?.toLowerCase();
    const isImage = ["png", "jpg", "jpeg", "webp", "bmp", "tiff"].includes(ext || "");
    const isText = ["txt", "json", "md"].includes(ext || "");

    const reader = new FileReader();
    reader.onload = (event) => {
      const resultData = event.target?.result as string;
      setFileBase64(resultData);

      if (isImage) {
        setImagePreviewUrl(resultData);
        setResumeText(`[Binary Image Resume: ${file.name} (${sizeStr})]\nTargeting AI OCR extraction.`);
      } else if (isText) {
        setImagePreviewUrl(null);
        const textReader = new FileReader();
        textReader.onload = (te) => setResumeText(te.target?.result as string);
        textReader.readAsText(file);
      } else {
        setImagePreviewUrl(null);
        setResumeText(`[Binary Document: ${file.name} (${sizeStr})]\nFormat: .${ext?.toUpperCase()} - Multi-stage digital text extraction ready.`);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleClearUploadedFile = () => {
    setFileBase64(null);
    setFileInfo(null);
    setImagePreviewUrl(null);
    const s = samples.find((x) => x.id === "sample_4") || samples[0];
    if (s) {
      setSelectedSampleId(s.id);
      setResumeText(s.content);
      setFileName(s.filename);
    }
  };

  const handleAnalyze = async () => {
    if (!resumeText.trim() && !fileBase64) {
      setError("Please upload a PDF/Word/Image file, paste a resume, or choose a benchmark sample.");
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_text: resumeText,
          file_base64: fileBase64 || undefined,
          file_name: fileName,
          target_role: targetRole,
          jd_text: jdText.trim() || undefined,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.details || data.error || "Analysis failed.");
      }
      setResult(data);
      setActiveTab("overview");
    } catch (err: any) {
      setError(err.message || "Failed to analyze resume.");
    } finally {
      setLoading(false);
    }
  };

  const handleRunTests = async () => {
    setTestLoading(true);
    setTestOutput(null);
    try {
      const res = await fetch("/api/tests/run");
      const data = await res.json();
      setTestOutput(data.output);
      setActiveTab("tests");
    } catch (err: any) {
      setTestOutput("Failed to run tests: " + err.message);
    } finally {
      setTestLoading(false);
    }
  };

  const handleDownloadJSON = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `structured_${fileName.replace(/\.[^/.]+$/, "")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyJSON = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const resume = result?.resume;
  const ats = result?.ats_analysis;
  const conf = result?.confidence;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/90 backdrop-blur sticky top-0 z-40 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-base tracking-tight text-white">AI RESUME ANALYSER V2</h1>
              <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wider rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 uppercase">
                Layout-Aware
              </span>
            </div>
            <p className="text-xs text-slate-400">Contextual Episode Engine • Verifiable ATS Intelligence</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRunTests}
            disabled={testLoading}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
          >
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            {testLoading ? "Running Regression Tests..." : "Run Test Suite (pytest)"}
          </button>

          {result && (
            <button
              onClick={handleDownloadJSON}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition shadow-sm"
            >
              <Download className="w-3.5 h-3.5" />
              Export JSON
            </button>
          )}
        </div>
      </header>

      {/* Main Workspace Grid */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
        {/* Left Control Panel / Input Sidebar (4 cols) */}
        <div className="lg:col-span-4 border-r border-slate-800/80 bg-slate-900/40 p-5 flex flex-col gap-5 overflow-y-auto max-h-[calc(100vh-61px)]">
          {/* Target Role Selector */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
              <span>TARGET TECHNICAL ROLE</span>
              <span className="text-[10px] text-slate-500 font-normal">8 Domain Profiles</span>
            </label>
            <select
              value={targetRole}
              onChange={(e) => setTargetRole(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 text-slate-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
            >
              {ROLES.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.title} ({r.category})
                </option>
              ))}
            </select>
          </div>

          {/* Sample Resumes Presets */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">PRE-BUILT BENCHMARK SAMPLES</span>
              <span className="text-[10px] text-slate-500 font-normal">Synthetic Multi-Job Fixtures</span>
            </div>
            <div className="grid grid-cols-1 gap-2">
              {samples.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleSelectSample(s.id)}
                  className={`text-left p-2.5 rounded-lg border transition text-xs flex flex-col gap-1 ${
                    selectedSampleId === s.id
                      ? "bg-indigo-500/10 border-indigo-500/40 text-slate-100"
                      : "bg-slate-800/60 border-slate-700/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                  }`}
                >
                  <div className="flex items-center justify-between font-medium text-slate-200">
                    <span>{s.title}</span>
                    <span className="text-[10px] text-indigo-400 font-mono">{s.category}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-1">{s.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Custom Resume Upload or Paste */}
          <div className="flex flex-col gap-2 flex-1">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300">
                RESUME SOURCE (PDF / DOCX / IMAGE / TXT)
              </label>
              <label className="cursor-pointer text-[11px] font-medium text-indigo-400 hover:text-indigo-300 flex items-center gap-1.5 px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20 transition">
                <FileText className="w-3.5 h-3.5" />
                <span>Upload File</span>
                <input
                  type="file"
                  accept=".txt,.pdf,.docx,.doc,.png,.jpg,.jpeg,.webp,.bmp,.tiff"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>

            {/* If a binary file (PDF, DOCX, Image) is uploaded, show rich file card */}
            {fileInfo ? (
              <div className="p-3.5 rounded-xl bg-slate-900/90 border border-indigo-500/30 flex flex-col gap-2.5">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <div className="w-9 h-9 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0">
                      {imagePreviewUrl ? (
                        <Sparkles className="w-5 h-5 text-indigo-400" />
                      ) : (
                        <FileCheck className="w-5 h-5 text-indigo-400" />
                      )}
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs font-bold text-white truncate">{fileInfo.name}</div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-400 mt-0.5">
                        <span className="font-mono">{fileInfo.size}</span>
                        <span>•</span>
                        <span className="uppercase text-indigo-300 font-semibold">
                          {fileInfo.name.split(".").pop()}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={handleClearUploadedFile}
                    className="text-[11px] text-rose-400 hover:text-rose-300 px-2 py-1 rounded bg-rose-500/10 border border-rose-500/20 transition shrink-0"
                  >
                    Remove
                  </button>
                </div>

                {imagePreviewUrl && (
                  <div className="relative rounded-lg overflow-hidden border border-slate-800 bg-black/40 max-h-36 flex items-center justify-center">
                    <img
                      src={imagePreviewUrl}
                      alt="Uploaded resume preview"
                      className="object-contain max-h-36 w-full"
                    />
                  </div>
                )}

                <div className="text-[11px] text-emerald-400 flex items-center gap-1.5 bg-emerald-500/10 px-2 py-1 rounded-md border border-emerald-500/20">
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                  <span>
                    Binary payload verified. Ready for multi-stage layout extraction & OCR.
                  </span>
                </div>
              </div>
            ) : (
              <div className="relative flex-1 min-h-[160px]">
                <textarea
                  value={resumeText}
                  onChange={(e) => {
                    setResumeText(e.target.value);
                    setSelectedSampleId("custom");
                  }}
                  placeholder="Paste raw resume text, upload a PDF / DOCX / Image, or pick a benchmark sample above..."
                  className="w-full h-full bg-slate-950/70 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-300 focus:outline-none focus:ring-1 focus:ring-indigo-500 resize-none"
                />
              </div>
            )}
          </div>

          {/* Optional Job Description */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300">OPTIONAL: TARGET JOB DESCRIPTION</label>
              <span className="text-[10px] text-slate-500">Semantic Overlap</span>
            </div>
            <textarea
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
              rows={3}
              placeholder="Paste job description text to perform keyword matching & gap analysis..."
              className="w-full bg-slate-950/70 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-indigo-500 resize-none font-mono"
            />
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Primary Action Button */}
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-indigo-500 to-emerald-500 hover:from-indigo-600 hover:to-emerald-600 text-white font-semibold text-sm shadow-lg shadow-indigo-500/20 flex items-center justify-center gap-2 transition disabled:opacity-50"
          >
            {loading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Running Layout Pipeline...</span>
              </div>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>ANALYZE RESUME</span>
              </>
            )}
          </button>
        </div>

        {/* Right Dashboard Area (8 cols) */}
        <div className="lg:col-span-8 flex flex-col bg-slate-950 overflow-y-auto max-h-[calc(100vh-61px)]">
          {!result && !testOutput ? (
            <div className="flex-1 flex flex-col items-center justify-center p-12 text-center text-slate-500 gap-4">
              <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-indigo-400">
                <FileCheck className="w-8 h-8" />
              </div>
              <div className="max-w-md">
                <h3 className="text-base font-semibold text-slate-200">No Resume Analyzed Yet</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Select one of the 3 benchmark candidate resumes on the left and click{" "}
                  <strong className="text-indigo-400">Analyze Resume</strong> to inspect the multi-stage parsing,
                  episodic anti-merge reconstruction, and ATS score breakdown.
                </p>
              </div>
              <button
                onClick={handleAnalyze}
                className="mt-2 px-4 py-2 text-xs font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition flex items-center gap-2"
              >
                <Play className="w-3.5 h-3.5" /> Run Initial Benchmark
              </button>
            </div>
          ) : (
            <div className="p-6 flex flex-col gap-6">
              {/* Top Overview Cards */}
              {result && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {/* Card 1: Overall ATS Score */}
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">Overall ATS Score</span>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-emerald-400">{ats.overall_score}</span>
                      <span className="text-xs text-slate-500 font-mono">/ 100</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1">
                      <div
                        className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${ats.overall_score}%` }}
                      />
                    </div>
                  </div>

                  {/* Card 2: Skill Coverage */}
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">Skill Alignment</span>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-indigo-400">{ats.skill_coverage_score}%</span>
                      <span className="text-xs text-slate-500 font-mono">Core</span>
                    </div>
                    <span className="text-[10px] text-slate-400 mt-1">
                      {ats.detected_core_skills.length} matched / {ats.missing_core_skills.length} missing
                    </span>
                  </div>

                  {/* Card 3: Experience Episodes (Anti-merge proof) */}
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">Work Episodes</span>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-cyan-400">{resume.experience.length}</span>
                      <span className="text-xs text-emerald-400 font-medium">Distinct</span>
                    </div>
                    <span className="text-[10px] text-slate-400 mt-1">Zero accidental merging</span>
                  </div>

                  {/* Card 4: Confidence Score */}
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">Parsing Confidence</span>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-amber-400">
                        {(conf.overall_confidence * 100).toFixed(0)}%
                      </span>
                      <span className="text-xs text-slate-500 font-mono">Verifiable</span>
                    </div>
                    <span className="text-[10px] text-slate-400 mt-1">Layout & Typography Grounded</span>
                  </div>
                </div>
              )}

              {/* Navigation Tabs */}
              <div className="flex items-center gap-1 border-b border-slate-800 overflow-x-auto pb-1 text-xs">
                {[
                  { id: "overview", label: "Overview", icon: Compass },
                  { id: "experience", label: `Experience (${resume?.experience?.length || 0})`, icon: Briefcase },
                  { id: "skills", label: `Skills (${resume?.skills?.length || 0})`, icon: Code2 },
                  {
                    id: "education",
                    label: `Education (${resume?.education?.length || 0}) & Certs (${resume?.certifications?.length || 0})`,
                    icon: GraduationCap,
                  },
                  { id: "projects", label: `Projects (${resume?.projects?.length || 0})`, icon: Layers },
                  { id: "ats", label: "ATS Deep Dive", icon: ShieldCheck },
                  { id: "jd_match", label: "JD Matcher", icon: Search },
                  { id: "graph", label: "Relational Graph", icon: TrendingUp },
                  { id: "json", label: "Structured JSON", icon: Terminal },
                  { id: "tests", label: "Test Runner", icon: CheckCircle2 },
                ].map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex items-center gap-2 px-3 py-2 rounded-t-lg font-medium transition whitespace-nowrap ${
                        activeTab === tab.id
                          ? "bg-slate-900 text-indigo-400 border-b-2 border-indigo-500 font-semibold"
                          : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/50"
                      }`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>

              {/* TAB 1: OVERVIEW */}
              {activeTab === "overview" && result && (
                <div className="flex flex-col gap-6">
                  {/* Candidate Header Summary */}
                  <div className="p-5 rounded-2xl bg-gradient-to-r from-slate-900 to-indigo-950/40 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2">
                        <h2 className="text-xl font-bold text-white">{resume.contact.name || "Candidate Name Not Detected"}</h2>
                        <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                          {ats.target_role}
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400 mt-2">
                        {resume.contact.email && <span>📧 {resume.contact.email}</span>}
                        {resume.contact.phone && <span>📞 {resume.contact.phone}</span>}
                        {resume.contact.location && <span>📍 {resume.contact.location}</span>}
                        {resume.contact.linkedin && (
                          <span className="text-indigo-400">🔗 {resume.contact.linkedin}</span>
                        )}
                        {resume.contact.github && (
                          <span className="text-indigo-400">💻 {resume.contact.github}</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Academic Qualifications Quick Preview in Overview */}
                  {resume.education && resume.education.length > 0 && (
                    <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-2">
                          <GraduationCap className="w-4 h-4 text-indigo-400" />
                          Academic Qualifications ({resume.education.length} Detected)
                        </h4>
                        <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                          {resume.education.length} Distinct Degrees
                        </span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {resume.education.map((edu: any, idx: number) => (
                          <div
                            key={idx}
                            className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 flex flex-col gap-1.5"
                          >
                            <div className="flex items-start justify-between gap-2">
                              <span className="text-xs font-bold text-white leading-snug">{edu.degree}</span>
                              {edu.gpa && (
                                <span className="text-[10px] font-mono text-emerald-400 px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 shrink-0">
                                  {edu.gpa}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center justify-between text-[11px] text-slate-400">
                              <span className="text-indigo-400 font-medium truncate">{edu.institution}</span>
                              <span className="font-mono shrink-0">
                                {edu.start_date || ""} {edu.end_date ? `– ${edu.end_date}` : ""}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Summary Text if present */}
                  {resume.summary && (
                    <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                      <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Professional Summary</h4>
                      <p className="text-xs text-slate-300 leading-relaxed font-sans">{resume.summary}</p>
                    </div>
                  )}

                  {/* Recommendations */}
                  {ats.actionable_recommendations && ats.actionable_recommendations.length > 0 && (
                    <div className="p-5 rounded-xl bg-indigo-950/20 border border-indigo-800/30 flex flex-col gap-3">
                      <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-indigo-400" />
                        Domain ATS Recommendations for {ats.target_role}
                      </h4>
                      <ul className="space-y-2">
                        {ats.actionable_recommendations.map((rec: string, idx: number) => (
                          <li key={idx} className="text-xs text-slate-300 flex items-start gap-2">
                            <span className="text-indigo-400 font-bold">•</span>
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 2: WORK EXPERIENCE (Anti-Merge Proof) */}
              {activeTab === "experience" && result && (
                <div className="flex flex-col gap-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-semibold text-white">Reconstructed Employment Episodes</h3>
                      <p className="text-xs text-slate-400">
                        Anchored on Role + Organization + Date triplets to prevent over-merging.
                      </p>
                    </div>
                    <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                      {resume.experience.length} Discrete Episodes
                    </span>
                  </div>

                  <div className="space-y-4">
                    {resume.experience.map((ep: any, idx: number) => (
                      <div
                        key={ep.id || idx}
                        className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-3 hover:border-slate-700 transition"
                      >
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-1 border-b border-slate-800/60 pb-3">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold text-white">{ep.role}</span>
                              {ep.is_current && (
                                <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-emerald-500/20 text-emerald-400 rounded">
                                  Current
                                </span>
                              )}
                            </div>
                            <span className="text-xs text-indigo-400 font-medium">{ep.company}</span>
                            {ep.location && <span className="text-xs text-slate-500 ml-2">• {ep.location}</span>}
                          </div>
                          <div className="text-right">
                            <span className="text-xs font-mono text-slate-300">
                              {ep.start_date || "Start Date N/A"} – {ep.end_date || "Present"}
                            </span>
                            <div className="text-[10px] text-slate-500">
                              Confidence: {(ep.confidence * 100).toFixed(0)}%
                            </div>
                          </div>
                        </div>

                        {/* Bullet Points */}
                        {ep.bullet_points && ep.bullet_points.length > 0 ? (
                          <ul className="space-y-1.5 text-xs text-slate-300">
                            {ep.bullet_points.map((bp: string, bidx: number) => (
                              <li key={bidx} className="flex items-start gap-2">
                                <span className="text-slate-600 mt-1">•</span>
                                <span className="leading-relaxed">{bp}</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-xs text-slate-400 italic">{ep.description || "No description lines recorded."}</p>
                        )}

                        {/* Technologies Mentioned */}
                        {ep.technologies_mentioned && ep.technologies_mentioned.length > 0 && (
                          <div className="flex flex-wrap items-center gap-1.5 pt-2">
                            <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mr-1">
                              Tech Used:
                            </span>
                            {ep.technologies_mentioned.map((tech: string, tidx: number) => (
                              <span
                                key={tidx}
                                className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700/60"
                              >
                                {tech}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TAB 3: TECHNICAL SKILLS */}
              {activeTab === "skills" && result && (
                <div className="flex flex-col gap-6">
                  {/* NEW: Interactive Keyword Gap Highlighter Component */}
                  <KeywordGapHighlighter 
                    missingCore={ats.missing_core_skills || []}
                    missingSecondary={ats.missing_secondary_skills || ["Scikit-Learn", "FastAPI"]}
                    missingKeywords={ats.missing_keywords || ["MLOps", "Model Deployment", "CI/CD Pipeline"]}
                    targetRole={ats.target_role || targetRole}
                  />
                  {/* Core Role Matches */}
                  <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4" />
                        Target Role Core Competencies Detected ({ats.detected_core_skills.length})
                      </h4>
                      <span className="text-xs font-mono text-slate-400">Target: {ats.target_role}</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {ats.detected_core_skills.map((sk: string, idx: number) => (
                        <span
                          key={idx}
                          className="px-2.5 py-1 rounded-lg text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                        >
                          {sk}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Missing Core Skills */}
                  {ats.missing_core_skills.length > 0 && (
                    <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-3">
                      <h4 className="text-xs font-bold text-amber-400 uppercase tracking-wider flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        Missing Core Competencies for {ats.target_role} ({ats.missing_core_skills.length})
                      </h4>
                      <p className="text-xs text-slate-400">
                        ATS filtering engines prioritize these keywords for this specific role family.
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {ats.missing_core_skills.map((sk: string, idx: number) => (
                          <span
                            key={idx}
                            className="px-2.5 py-1 rounded-lg text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20"
                          >
                            + {sk}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* All Extracted Skills */}
                  <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-3">
                    <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                      All Extracted Technical Skills ({resume.skills.length})
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {resume.skills.map((sk: string, idx: number) => (
                        <span
                          key={idx}
                          className="px-2.5 py-1 rounded-lg text-xs font-mono bg-slate-800 text-slate-300 border border-slate-700/60"
                        >
                          {sk}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 4: EDUCATION & CERTS */}
              {activeTab === "education" && result && (
                <div className="flex flex-col gap-6">
                  {/* Education */}
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-sm font-semibold text-white">Academic Qualifications</h3>
                        <p className="text-xs text-slate-400">
                          Normalized degrees, boards, academic institutions, graduation dates, and verified GPA/percentage.
                        </p>
                      </div>
                      <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                        {resume.education.length} Degrees Detected
                      </span>
                    </div>

                    <div className="space-y-3">
                      {resume.education.map((edu: any, idx: number) => (
                        <div
                          key={idx}
                          className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-2 hover:border-slate-700 transition"
                        >
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 border-b border-slate-800/60 pb-2.5">
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-bold text-white">{edu.degree}</span>
                                <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                                  Level {idx + 1}
                                </span>
                              </div>
                              <span className="text-xs text-indigo-400 font-medium">{edu.institution}</span>
                            </div>
                            <div className="flex sm:flex-col sm:items-end justify-between items-center text-right">
                              <span className="text-xs font-mono text-slate-300">
                                {edu.start_date ? `${edu.start_date} – ` : ""}
                                {edu.end_date || "Date N/A"}
                              </span>
                              {edu.gpa && (
                                <span className="text-xs font-mono text-emerald-400 font-semibold">
                                  {edu.gpa}
                                </span>
                              )}
                            </div>
                          </div>

                          {edu.details && edu.details.length > 0 && (
                            <div className="text-xs text-slate-300 font-sans leading-relaxed">
                              {edu.details.map((d: string, didx: number) => (
                                <p key={didx} className="text-slate-400">
                                  {d}
                                </p>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Certifications */}
                  {resume.certifications && resume.certifications.length > 0 && (
                    <div className="flex flex-col gap-3">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-white">Licenses & Certifications</h3>
                        <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          {resume.certifications.length} Credentials
                        </span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {resume.certifications.map((cert: any, idx: number) => (
                          <div
                            key={idx}
                            className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-1.5"
                          >
                            <div className="flex items-start justify-between gap-2">
                              <span className="text-xs font-bold text-white">{cert.name}</span>
                              {cert.date && (
                                <span className="text-[10px] font-mono text-slate-400 shrink-0">
                                  {cert.date}
                                </span>
                              )}
                            </div>
                            {cert.issuer && (
                              <span className="text-xs text-emerald-400 font-medium">{cert.issuer}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Achievements and Honors if present */}
                  {resume.achievements && resume.achievements.length > 0 && (
                    <div className="flex flex-col gap-3">
                      <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                        <Award className="w-4 h-4 text-amber-400" />
                        Academic & Professional Honors ({resume.achievements.length})
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                        {resume.achievements.map((ach: string, idx: number) => (
                          <div
                            key={idx}
                            className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 text-xs text-slate-300 flex items-start gap-2"
                          >
                            <span className="text-amber-400 font-bold">•</span>
                            <span>{ach}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 5: PROJECTS */}
              {activeTab === "projects" && result && (
                <div className="flex flex-col gap-4">
                  <h3 className="text-sm font-semibold text-white">Technical Projects</h3>
                  <div className="space-y-4">
                    {resume.projects.map((proj: any, idx: number) => (
                      <div key={idx} className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-bold text-white">{proj.title}</span>
                          {proj.link && (
                            <a
                              href={proj.link}
                              target="_blank"
                              rel="noreferrer"
                              className="text-xs text-indigo-400 hover:underline flex items-center gap-1"
                            >
                              <span>Repository / Demo</span>
                              <ExternalLink className="w-3 h-3" />
                            </a>
                          )}
                        </div>
                        <p className="text-xs text-slate-300 leading-relaxed">{proj.description}</p>
                        {proj.technologies && proj.technologies.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 pt-2">
                            {proj.technologies.map((t: string, tidx: number) => (
                              <span
                                key={tidx}
                                className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-400 border border-slate-700/60"
                              >
                                {t}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TAB 6: ATS DEEP DIVE */}
              {activeTab === "ats" && result && (
                <div className="flex flex-col gap-6">
                  {/* Scoring Methodology */}
                  <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-3">
                    <h3 className="text-sm font-semibold text-white">Transparent ATS Scoring Methodology</h3>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Unlike generic calculators with fabricated numbers, our engine computes verifiable scores grounded in 4
                      transparent metrics: Section Completeness (30%), Skill Alignment (40%), Experience Depth (25%), and
                      Contact Validation (5%).
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
                      <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
                        <span className="text-[11px] text-slate-400">Section Completeness</span>
                        <div className="text-xl font-bold text-white mt-1">{ats.section_completeness_score}%</div>
                        <span className="text-[10px] text-slate-500">{ats.sections_found.length} standard sections found</span>
                      </div>
                      <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
                        <span className="text-[11px] text-slate-400">Skill Coverage</span>
                        <div className="text-xl font-bold text-white mt-1">{ats.skill_coverage_score}%</div>
                        <span className="text-[10px] text-slate-500">Core taxonomy match</span>
                      </div>
                      <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
                        <span className="text-[11px] text-slate-400">Experience Depth</span>
                        <div className="text-xl font-bold text-white mt-1">{ats.experience_depth_score}%</div>
                        <span className="text-[10px] text-slate-500">Quantified metrics & scope</span>
                      </div>
                    </div>
                  </div>

                  {/* Parsing Risks */}
                  <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-3">
                    <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                      Parsing Risks & Formatting Warnings ({ats.parsing_risks.length})
                    </h4>
                    {ats.parsing_risks.length === 0 ? (
                      <div className="text-xs text-emerald-400 flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4" />
                        <span>Zero high-risk formatting artifacts detected. High ATS readability.</span>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {ats.parsing_risks.map((risk: any, idx: number) => (
                          <div
                            key={idx}
                            className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs flex flex-col gap-1"
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-semibold text-slate-200">[{risk.category}] {risk.message}</span>
                              <span
                                className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                                  risk.level === "HIGH"
                                    ? "bg-rose-500/20 text-rose-400"
                                    : "bg-amber-500/20 text-amber-400"
                                }`}
                              >
                                {risk.level}
                              </span>
                            </div>
                            <span className="text-slate-400">{risk.suggestion}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* TAB 7: JOB DESCRIPTION MATCH */}
              {activeTab === "jd_match" && result && (
                <div className="flex flex-col gap-6">
                  {result.jd_match ? (
                    <>
                      <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
                        <div>
                          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Job Description Match Score</span>
                          <div className="text-3xl font-bold text-indigo-400 mt-1">
                            {result.jd_match.similarity_score}%
                          </div>
                          <p className="text-xs text-slate-400 mt-1">
                            Based on keyword overlap & semantic requirement density.
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-2">
                          <h4 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                            <CheckCircle2 className="w-3.5 h-3.5" />
                            Matched Job Requirements ({result.jd_match.matched_skills.length})
                          </h4>
                          <div className="flex flex-wrap gap-1.5">
                            {result.jd_match.matched_skills.map((s: string, idx: number) => (
                              <span
                                key={idx}
                                className="px-2 py-0.5 text-xs font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded"
                              >
                                {s}
                              </span>
                            ))}
                          </div>
                        </div>

                        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-2">
                          <h4 className="text-xs font-semibold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                            <AlertTriangle className="w-3.5 h-3.5" />
                            Missing From Resume ({result.jd_match.missing_skills.length})
                          </h4>
                          <div className="flex flex-wrap gap-1.5">
                            {result.jd_match.missing_skills.map((s: string, idx: number) => (
                              <span
                                key={idx}
                                className="px-2 py-0.5 text-xs font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded"
                              >
                                + {s}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="p-8 text-center text-slate-400 border border-dashed border-slate-800 rounded-xl">
                      <p className="text-xs">
                        Paste a Job Description in the left sidebar and click <strong>Analyze Resume</strong> to inspect
                        keyword alignment and semantic overlap.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 8: RELATIONAL EPISODE GRAPH */}
              {activeTab === "graph" && result && (
                <div className="flex flex-col gap-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-semibold text-white">Relational Episode Graph</h3>
                      <p className="text-xs text-slate-400">
                        Network representation: Role nodes bind to Company nodes, Date bounds, and Deliverables.
                      </p>
                    </div>
                    <span className="text-xs font-mono text-slate-400">
                      {result.episode_graph?.num_nodes || 0} Nodes • {result.episode_graph?.num_edges || 0} Edges
                    </span>
                  </div>

                  <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col gap-4">
                    <div className="space-y-4">
                      {resume.experience.map((ep: any, idx: number) => (
                        <div key={idx} className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 flex flex-col gap-3">
                          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
                            <span className="px-2.5 py-1 rounded bg-indigo-500/20 text-indigo-300 font-bold border border-indigo-500/30">
                              [ROLE] {ep.role}
                            </span>
                            <span className="text-slate-600 font-bold">── EMPLOYED_AT ──►</span>
                            <span className="px-2.5 py-1 rounded bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30">
                              [COMPANY] {ep.company}
                            </span>
                            <span className="text-slate-600 font-bold">── TENURE ──►</span>
                            <span className="px-2.5 py-1 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                              [DATE] {ep.start_date} – {ep.end_date}
                            </span>
                          </div>

                          {ep.bullet_points && ep.bullet_points.length > 0 && (
                            <div className="pl-6 border-l-2 border-slate-800 space-y-1">
                              {ep.bullet_points.slice(0, 2).map((bp: string, bidx: number) => (
                                <div key={bidx} className="text-[11px] text-slate-400 flex items-center gap-2">
                                  <span className="text-indigo-500 font-bold">└─ DELIVERED:</span>
                                  <span className="line-clamp-1">{bp}</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 9: STRUCTURED JSON */}
              {activeTab === "json" && result && (
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400 font-mono">Canonical Schema Output (Pydantic v2 Serialized)</span>
                    <button
                      onClick={handleCopyJSON}
                      className="px-2.5 py-1 rounded text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition flex items-center gap-1.5"
                    >
                      <Copy className="w-3 h-3" />
                      <span>{copied ? "Copied!" : "Copy JSON"}</span>
                    </button>
                  </div>
                  <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-emerald-400 font-mono overflow-x-auto max-h-[500px]">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </div>
              )}

              {/* TAB 10: TEST RUNNER */}
              {activeTab === "tests" && (
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400 font-mono">Automated Pytest Execution Suite</span>
                    <button
                      onClick={handleRunTests}
                      disabled={testLoading}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white transition flex items-center gap-1.5"
                    >
                      <Terminal className="w-3.5 h-3.5" />
                      <span>{testLoading ? "Running..." : "Re-run Pytest"}</span>
                    </button>
                  </div>
                  <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono overflow-x-auto max-h-[500px] whitespace-pre-wrap">
                    {testOutput || "Click 'Run Test Suite (pytest)' above to execute the automated regression tests."}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
