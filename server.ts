import express from "express";
import { createServer as createViteServer } from "vite";
import fs from "fs";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import crypto from "crypto";

const execFileAsync = promisify(execFile);

const app = express();
const PORT = 3000;

// ============================================================
// PYTHON EXECUTABLE RESOLUTION
// ============================================================
// Automatically detects the Python executable from:
//
// Windows:
//   venv/Scripts/python.exe
//   .venv/Scripts/python.exe
//
// Linux / macOS / Colab:
//   venv/bin/python
//   .venv/bin/python
//
// If no virtual environment exists:
//   Windows      -> python
//   Linux/macOS -> python3
// ============================================================

const getPythonExecutable = (): string => {
  const isWin = process.platform === "win32";

  const venvNames = ["venv", ".venv"];

  for (const venvName of venvNames) {
    const pythonPath = isWin
      ? path.join(
          process.cwd(),
          venvName,
          "Scripts",
          "python.exe"
        )
      : path.join(
          process.cwd(),
          venvName,
          "bin",
          "python"
        );

    if (fs.existsSync(pythonPath)) {
      return pythonPath;
    }
  }

  // Fallback to system Python
  return isWin ? "python" : "python3";
};

// Detect Python once when the server starts
const pythonExecutable = getPythonExecutable();

console.log(
  `[AI Resume Analyser V2] Python executable: ${pythonExecutable}`
);

// ============================================================
// EXPRESS CONFIGURATION
// ============================================================

app.use(express.json({ limit: "25mb" }));

app.use(
  express.urlencoded({
    extended: true,
    limit: "25mb",
  })
);

// ============================================================
// 1. API: SUPPORTED TECHNICAL ROLES
// ============================================================

app.get("/api/roles", (req, res) => {
  try {
    const rolesPath = path.resolve(
      process.cwd(),
      "config/job_roles.yaml"
    );

    if (fs.existsSync(rolesPath)) {
      /*
       * The actual YAML file is used as the existence check.
       *
       * The frontend receives the canonical technical roles defined
       * for the current version of the application.
       */

      const roles = [
        {
          id: "data_scientist",
          title: "Data Scientist",
          category: "Data Science",
        },
        {
          id: "ml_engineer",
          title: "Machine Learning Engineer",
          category: "Machine Learning",
        },
        {
          id: "ai_engineer",
          title: "AI Engineer",
          category: "Artificial Intelligence",
        },
        {
          id: "genai_engineer",
          title: "Generative AI Engineer",
          category: "Generative AI",
        },
        {
          id: "data_engineer",
          title: "Data Engineer",
          category: "Data Engineering",
        },
        {
          id: "data_analyst",
          title: "Data Analyst",
          category: "Data Analytics",
        },
        {
          id: "mlops_engineer",
          title: "MLOps Engineer",
          category: "MLOps & Cloud",
        },
        {
          id: "software_engineer",
          title: "Software Engineer (Python)",
          category: "Software Engineering",
        },
      ];

      return res.json({
        roles,
      });
    }

    return res.json({
      roles: [],
    });
  } catch (err: any) {
    console.error("Roles API error:", err);

    return res.status(500).json({
      error: err.message,
    });
  }
});

// ============================================================
// 2. API: SAMPLE RESUMES
// ============================================================

app.get("/api/samples", (req, res) => {
  try {
    const samplesDir = path.resolve(
      process.cwd(),
      "data/samples"
    );

    const sampleDefinitions = [
      {
        id: "sample_4",
        filename: "sample_resume_4.txt",
        title: "AI Engineer & Data Scientist (John Doe)",
        category: "AI / Data Science (4 Degrees)",
        description:
          "4 academic qualifications (MSc, B.Tech, Class XII, Class X), 3 work episodes, 34 skills.",
      },
      {
        id: "sample_1",
        filename: "sample_resume_1.txt",
        title: "Senior Data Scientist (Alexander Chen)",
        category: "Data Science / ML",
        description:
          "5+ years experience, 3 distinct episodes (Nexus Analytics, Apex AI, ComplyHappy Intern).",
      },
      {
        id: "sample_2",
        filename: "sample_resume_2.txt",
        title: "Lead MLOps Engineer (Priya Sharma)",
        category: "MLOps & Infrastructure",
        description:
          "Kubernetes, MLflow, CI/CD, Prometheus drift monitoring, 2 distinct corporate roles.",
      },
      {
        id: "sample_3",
        filename: "sample_resume_3.txt",
        title: "Generative AI Engineer (Marcus Vance)",
        category: "Generative AI / LLM",
        description:
          "Enterprise RAG, LangChain, LangGraph agent workflows, Pinecone vector database.",
      },
    ];

    const samples = sampleDefinitions.map((sample) => {
      const samplePath = path.join(
        samplesDir,
        sample.filename
      );

      return {
        ...sample,
        content: fs.existsSync(samplePath)
          ? fs.readFileSync(samplePath, "utf-8")
          : "",
      };
    });

    return res.json({
      samples,
    });
  } catch (err: any) {
    console.error("Samples API error:", err);

    return res.status(500).json({
      error: err.message,
    });
  }
});

// ============================================================
// 3. API: ANALYZE RESUME
// ============================================================
//
// Supports:
// - PDF
// - DOCX
// - DOC
// - Images
// - TXT
// - Pasted resume text
//
// The Node/Express server creates a temporary file and invokes
// the Python resume-analysis pipeline.
// ============================================================

app.post("/api/analyze", async (req, res) => {
  let requestDir = "";

  try {
    const {
      resume_text,
      file_base64,
      file_name,
      target_role = "data_scientist",
      jd_text,
    } = req.body ?? {};

    const hasText =
      typeof resume_text === "string" &&
      resume_text.trim().length > 0;

    const hasFile =
      typeof file_base64 === "string" &&
      file_base64.trim().length > 0;

    if (!hasText && !hasFile) {
      return res.status(400).json({
        error:
          "Missing resume content. Upload a supported document/image or paste resume text.",
      });
    }

    const allowedRoles = new Set([
      "data_scientist",
      "ml_engineer",
      "ai_engineer",
      "genai_engineer",
      "data_engineer",
      "data_analyst",
      "mlops_engineer",
      "software_engineer",
    ]);

    if (typeof target_role !== "string" || !allowedRoles.has(target_role)) {
      return res.status(400).json({
        error: "Unsupported target role.",
      });
    }

    if (hasText && resume_text.length > 2_000_000) {
      return res.status(413).json({
        error: "Pasted resume text exceeds the 2 MB limit.",
      });
    }

    if (typeof jd_text === "string" && jd_text.length > 200_000) {
      return res.status(413).json({
        error: "Job description exceeds the 200 KB limit.",
      });
    }

    const tmpRoot = path.resolve(process.cwd(), "data/outputs/temp");
    fs.mkdirSync(tmpRoot, { recursive: true });
    requestDir = fs.mkdtempSync(path.join(tmpRoot, "request-"));

    let safeExt = ".txt";
    let resumeBuffer: Buffer;

    if (hasFile) {
      const originalExt = path.extname(
        typeof file_name === "string" ? file_name : ""
      ).toLowerCase();

      const allowedExtensions = new Set([
        ".pdf",
        ".docx",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".txt",
      ]);

      if (originalExt === ".doc") {
        return res.status(400).json({
          error: "Legacy .doc files are not supported. Convert the file to .docx or PDF.",
        });
      }

      if (originalExt && !allowedExtensions.has(originalExt)) {
        return res.status(400).json({
          error: `Unsupported file type: ${originalExt}`,
        });
      }

      const cleanBase64 = file_base64.replace(
        /^data:[^;]+;base64,/i,
        ""
      );

      try {
        resumeBuffer = Buffer.from(cleanBase64, "base64");
      } catch {
        return res.status(400).json({ error: "Invalid base64 upload." });
      }

      if (!resumeBuffer.length) {
        return res.status(400).json({ error: "Uploaded file is empty." });
      }

      const maxUploadBytes = 15 * 1024 * 1024;
      if (resumeBuffer.length > maxUploadBytes) {
        return res.status(413).json({
          error: "Uploaded resume exceeds the 15 MB limit.",
        });
      }

      const inferredExt = inferExtensionFromBytes(resumeBuffer);
      safeExt = originalExt || inferredExt || ".txt";

      if (safeExt !== ".txt" && inferredExt && safeExt !== inferredExt) {
        // DOCX is a ZIP container, so byte detection intentionally maps to .docx.
        // For real binary formats, a claimed extension must agree with the bytes.
        return res.status(400).json({
          error: "File extension does not match the uploaded file contents.",
        });
      }

      if (safeExt !== ".txt" && !inferredExt) {
        return res.status(400).json({
          error: "Could not verify the uploaded file type.",
        });
      }
    } else {
      resumeBuffer = Buffer.from(resume_text, "utf8");
      safeExt = ".txt";
    }

    const resumePath = path.join(
      requestDir,
      `input-${crypto.randomUUID()}${safeExt}`
    );
    const outputPath = path.join(
      requestDir,
      `output-${crypto.randomUUID()}.json`
    );

    fs.writeFileSync(resumePath, resumeBuffer);

    const args: string[] = [
      "run.py",
      resumePath,
      "--role",
      target_role,
    ];

    if (typeof jd_text === "string" && jd_text.trim().length > 10) {
      const jdPath = path.join(
        requestDir,
        `jd-${crypto.randomUUID()}.txt`
      );
      fs.writeFileSync(jdPath, jd_text, "utf8");
      args.push("--jd", jdPath);
    }

    args.push("--output", outputPath);

    const python = getPythonExecutable();

    console.log(
      `[AI Resume Analyser V2] Using Python: ${python}`
    );

    const { stdout, stderr } = await execFileAsync(
      python,
      args,
      {
        cwd: process.cwd(),
        timeout: 60_000,
        maxBuffer: 50 * 1024 * 1024,
        windowsHide: true,
      }
    );

    if (stderr?.trim()) {
      console.warn(
        "[AI Resume Analyser V2] Python stderr:",
        stderr.trim()
      );
    }

    if (!fs.existsSync(outputPath)) {
      throw new Error(
        "Pipeline completed without generating the expected JSON output."
      );
    }

    const rawJson = fs.readFileSync(outputPath, "utf8");
    const parsedResult = JSON.parse(rawJson);

    return res.json(parsedResult);
  } catch (err: any) {
    console.error(
      "[AI Resume Analyser V2] Analyze API error:",
      err
    );

    const details =
      err?.stderr ||
      err?.message ||
      "Unknown pipeline error.";

    const status =
      err?.code === "ETIMEDOUT" || err?.killed
        ? 504
        : 500;

    return res.status(status).json({
      error:
        status === 504
          ? "Resume analysis timed out."
          : "Resume analysis failed.",
      details,
    });
  } finally {
    // Every request gets its own isolated temporary directory. Always clean
    // it up so uploaded resumes and generated JSON do not accumulate.
    if (requestDir && fs.existsSync(requestDir)) {
      try {
        fs.rmSync(requestDir, {
          recursive: true,
          force: true,
        });
      } catch (cleanupErr) {
        console.warn(
          "[AI Resume Analyser V2] Temporary-file cleanup failed:",
          cleanupErr
        );
      }
    }
  }
});

// ============================================================
// FILE SIGNATURE VALIDATION
// ============================================================

function inferExtensionFromBytes(buffer: Buffer): string | null {
  if (buffer.length >= 5 && buffer.subarray(0, 5).toString() === "%PDF-") {
    return ".pdf";
  }

  if (
    buffer.length >= 8 &&
    buffer.subarray(0, 8).equals(
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    )
  ) {
    return ".png";
  }

  if (
    buffer.length >= 3 &&
    buffer.subarray(0, 3).equals(Buffer.from([0xff, 0xd8, 0xff]))
  ) {
    return ".jpg";
  }

  if (
    buffer.length >= 12 &&
    buffer.subarray(0, 4).toString() === "RIFF" &&
    buffer.subarray(8, 12).toString() === "WEBP"
  ) {
    return ".webp";
  }

  if (buffer.length >= 2 && buffer.subarray(0, 2).toString() === "BM") {
    return ".bmp";
  }

  if (
    buffer.length >= 4 &&
    (
      buffer.subarray(0, 4).equals(Buffer.from([0x49, 0x49, 0x2a, 0x00])) ||
      buffer.subarray(0, 4).equals(Buffer.from([0x4d, 0x4d, 0x00, 0x2a]))
    )
  ) {
    return ".tiff";
  }

  // DOCX is an OOXML ZIP container.
  if (
    buffer.length >= 4 &&
    buffer.subarray(0, 4).equals(Buffer.from([0x50, 0x4b, 0x03, 0x04]))
  ) {
    return ".docx";
  }

  // Plain text has no reliable magic signature; it is accepted when the
  // client explicitly sends .txt or when resume text is pasted.
  return null;
}

// ============================================================
// 4. API: RUN PYTEST
// ============================================================
//
// IMPORTANT:
// Instead of calling:
//
//     pytest tests/ -v
//
// we use:
//
//     python -m pytest tests/ -v
//
// using the same Python environment as the application.
// ============================================================

app.get("/api/tests/run", async (req, res) => {
  if (process.env.NODE_ENV === "production") {
    return res.status(404).json({ error: "Not found" });
  }

  try {
    const python = getPythonExecutable();

    const { stdout, stderr } = await execFileAsync(
      python,
      ["-m", "pytest", "tests/", "-v"],
      {
        cwd: process.cwd(),
        timeout: 120000,
        maxBuffer: 50 * 1024 * 1024,
        windowsHide: true,
      }
    );

    return res.json({
      success: true,
      output: stdout || stderr,
    });
  } catch (err: any) {
    console.error(
      "[AI Resume Analyser V2] Pytest error:",
      err
    );

    return res.json({
      success: false,
      output:
        err.stdout ||
        err.stderr ||
        err.message,
    });
  }
});

// ============================================================
// 5. SERVER STARTUP
// ============================================================

async function startServer() {
  const isProd =
    process.env.NODE_ENV === "production";

  // ----------------------------------------------------------
  // Development mode
  // ----------------------------------------------------------

  if (!isProd) {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
      },
      appType: "spa",
    });

    app.use(vite.middlewares);
  }

  // ----------------------------------------------------------
  // Production mode
  // ----------------------------------------------------------

  else {
    const distPath = path.resolve(
      process.cwd(),
      "dist"
    );

    app.use(
      express.static(distPath)
    );

    app.get("*", (req, res) => {
      res.sendFile(
        path.resolve(
          distPath,
          "index.html"
        )
      );
    });
  }

  // ----------------------------------------------------------
  // Start Express server
  // ----------------------------------------------------------

  app.listen(
    PORT,
    "0.0.0.0",
    () => {
      console.log(
        `[AI Resume Analyser V2] Server running on http://0.0.0.0:${PORT}`
      );

      console.log(
        `[AI Resume Analyser V2] Python: ${pythonExecutable}`
      );

      console.log(
        `[AI Resume Analyser V2] Environment: ${
          isProd
            ? "production"
            : "development"
        }`
      );
    }
  );
}

// ============================================================
// START APPLICATION
// ============================================================

startServer().catch((err) => {
  console.error(
    "[AI Resume Analyser V2] Failed to start server:",
    err
  );

  process.exit(1);
});