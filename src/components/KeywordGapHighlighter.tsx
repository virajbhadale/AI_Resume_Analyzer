import React, { useState } from "react";
import { AlertTriangle, CheckCircle2, Sparkles, PlusCircle } from "lucide-react";

interface KeywordGapProps {
  missingCore: string[];
  missingSecondary: string[];
  missingKeywords: string[];
  targetRole: string;
}

export default function KeywordGapHighlighter({
  missingCore = [],
  missingSecondary = [],
  missingKeywords = [],
  targetRole = "",
}: KeywordGapProps) {
  const [selectedCategory, setSelectedCategory] = useState<"core" | "secondary" | "keywords">("core");

  return (
    <div className="p-5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4 shadow-xl">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            Missing Technical Keywords & Skill Gaps
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Target Role: <span className="text-indigo-400 font-semibold uppercase">{targetRole || "General"}</span>
          </p>
        </div>
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setSelectedCategory("core")}
            className={`px-3 py-1.5 rounded-md transition ${selectedCategory === "core" ? "bg-amber-500/20 text-amber-300 font-semibold" : "text-slate-400 hover:text-slate-200"}`}
          >
            Core ({missingCore.length})
          </button>
          <button
            onClick={() => setSelectedCategory("secondary")}
            className={`px-3 py-1.5 rounded-md transition ${selectedCategory === "secondary" ? "bg-indigo-500/20 text-indigo-300 font-semibold" : "text-slate-400 hover:text-slate-200"}`}
          >
            Secondary ({missingSecondary.length})
          </button>
          <button
            onClick={() => setSelectedCategory("keywords")}
            className={`px-3 py-1.5 rounded-md transition ${selectedCategory === "keywords" ? "bg-cyan-500/20 text-cyan-300 font-semibold" : "text-slate-400 hover:text-slate-200"}`}
          >
            Keywords ({missingKeywords.length})
          </button>
        </div>
      </div>

      {/* Keyword Chips Grid Container */}
      <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 min-h-[130px] flex flex-wrap gap-2 items-start content-start">
        {selectedCategory === "core" &&
          (missingCore.length === 0 ? (
            <p className="text-xs text-emerald-400 flex items-center gap-1.5 py-2">
              <CheckCircle2 className="w-4 h-4" /> All core technical skills detected!
            </p>
          ) : (
            missingCore.map((kw, idx) => (
              <span key={idx} className="px-2.5 py-1 rounded-md text-xs font-mono bg-amber-500/10 text-amber-400 border border-amber-500/30 flex items-center gap-1.5 shadow-sm">
                <PlusCircle className="w-3 h-3 text-amber-400" /> {kw}
              </span>
            ))
          ))}

        {selectedCategory === "secondary" &&
          (missingSecondary.length === 0 ? (
            <p className="text-xs text-emerald-400 flex items-center gap-1.5 py-2">
              <CheckCircle2 className="w-4 h-4" /> All secondary stack technologies present!
            </p>
          ) : (
            missingSecondary.map((kw, idx) => (
              <span key={idx} className="px-2.5 py-1 rounded-md text-xs font-mono bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 flex items-center gap-1.5 shadow-sm">
                <PlusCircle className="w-3 h-3 text-indigo-400" /> {kw}
              </span>
            ))
          ))}

        {selectedCategory === "keywords" &&
          (missingKeywords.length === 0 ? (
            <p className="text-xs text-emerald-400 flex items-center gap-1.5 py-2">
              <CheckCircle2 className="w-4 h-4" /> Excellent semantic keyword coverage!
            </p>
          ) : (
            missingKeywords.map((kw, idx) => (
              <span key={idx} className="px-2.5 py-1 rounded-md text-xs font-mono bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 flex items-center gap-1.5 shadow-sm">
                <Sparkles className="w-3 h-3 text-cyan-400" /> {kw}
              </span>
            ))
          ))}
      </div>
    </div>
  );
}