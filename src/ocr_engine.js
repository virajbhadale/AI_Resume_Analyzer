import fs from "fs";
import path from "path";
import { GoogleGenAI } from "@google/genai";

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("GEMINI_API_KEY is not set.");
  process.exit(1);
}

const ai = new GoogleGenAI({
  apiKey,
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    },
  },
});

async function ocrImage(filePath) {
  try {
    const ext = path.extname(filePath).toLowerCase();
    let mimeType = "image/png";
    if (ext === ".jpg" || ext === ".jpeg") mimeType = "image/jpeg";
    else if (ext === ".webp") mimeType = "image/webp";

    const imageBuffer = fs.readFileSync(filePath);
    const base64Data = imageBuffer.toString("base64");

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: {
        parts: [
          {
            inlineData: {
              data: base64Data,
              mimeType: mimeType,
            },
          },
          {
            text: `Extract all text from this resume image accurately and in logical resume reading order.

Rules:
1. Preserve the candidate name, contact details, section headings, dates, company names, job titles, bullets, project titles, technologies, URLs, and other meaningful text verbatim whenever readable.
2. Preserve meaningful line breaks. Keep each bullet on its own line.
3. For two-column or sidebar layouts, finish the left/sidebar column from top to bottom before continuing with the main column, unless a full-width header clearly spans both columns. Never interleave unrelated left/right lines merely because they share the same vertical position.
4. Do not invent missing text. If a character is genuinely unreadable, preserve the readable portion rather than guessing.
5. Output ONLY the extracted resume text. Do not add explanations, confidence commentary, markdown, or code fences.`,
          },
        ],
      },
    });

    const text = response.text || "";
    // Clean code block fences if any
    const cleaned = text.replace(/^```[a-z]*\n/i, "").replace(/\n```$/, "").trim();
    return cleaned;
  } catch (err) {
    console.error("OCR Error:", err);
    throw err;
  }
}

async function main() {
  const filePath = process.argv[2];
  if (!filePath) {
    console.error("Usage: node ocr_engine.js <image_path>");
    process.exit(1);
  }
  const result = await ocrImage(filePath);
  console.log(result);
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  main();
}

export { ocrImage };
