const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);
const DEFAULT_POLL_INTERVAL_MS = Number(import.meta.env.VITE_ANALYSIS_POLL_MS || 2000);
const DEFAULT_TIMEOUT_MS = Number(import.meta.env.VITE_ANALYSIS_TIMEOUT_MS || 15 * 60 * 1000);
const ALLOW_DIRECT_FALLBACK =
  String(import.meta.env.VITE_ALLOW_DIRECT_FALLBACK || "false").toLowerCase() === "true";

export type FrameResult = {
  timestamp: number;
  fake_prob: number;
  fake_prob_smooth?: number;
  av_mismatch?: number;
  causal_breach_score?: number;
  scm_violation?: boolean;
  scm_z?: number;
  bbox?: [number, number, number, number] | null;
};

export type AnalyzeResponse = {
  video_fake: string | number;
  fake_confidence?: number;
  overall_score?: number;
  causal_breach_score?: number;
  scm_enabled?: boolean;
  highlight_timestamps?: number[];
  causal_segments?: { start: number; end: number; score?: number }[];
  frames: FrameResult[];
};

type AnalyzeAsyncSubmitResponse = {
  job_id: string;
  status: "queued" | "running";
};

type AnalyzeJobStatusResponse = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  result?: AnalyzeResponse;
  error?: string | null;
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function runDirectAnalysis(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || "Backend analysis failed");
  }

  return res.json();
}

async function submitAsyncAnalysis(file: File): Promise<AnalyzeAsyncSubmitResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/analyze/async`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || "Async submission failed");
  }

  return res.json();
}

async function fetchAsyncStatus(jobId: string): Promise<AnalyzeJobStatusResponse> {
  const res = await fetch(`${API_URL}/analyze/status/${jobId}`);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || "Async status check failed");
  }
  return res.json();
}

export async function analyzeVideo(
  file: File,
  options: { pollIntervalMs?: number; timeoutMs?: number } = {},
): Promise<AnalyzeResponse> {
  const {
    pollIntervalMs = Number.isFinite(DEFAULT_POLL_INTERVAL_MS) ? DEFAULT_POLL_INTERVAL_MS : 2000,
    timeoutMs = Number.isFinite(DEFAULT_TIMEOUT_MS) ? DEFAULT_TIMEOUT_MS : 15 * 60 * 1000,
  } = options;

  try {
    const submit = await submitAsyncAnalysis(file);
    const start = Date.now();

    while (true) {
      if (Date.now() - start > timeoutMs) {
        throw new Error(`Analysis timed out after ${Math.round(timeoutMs / 1000)}s. Please try a shorter video.`);
      }

      const status = await fetchAsyncStatus(submit.job_id);
      if (status.status === "completed" && status.result) {
        return status.result;
      }
      if (status.status === "failed") {
        throw new Error(status.error || "Analysis failed on the server.");
      }

      await sleep(pollIntervalMs);
    }
  } catch (error: any) {
    if (!ALLOW_DIRECT_FALLBACK) {
      throw error;
    }
    const message = error?.message || "";
    if (message.includes("404") || message.includes("405")) {
      return runDirectAnalysis(file);
    }
    if (message.toLowerCase().includes("async submission failed")) {
      return runDirectAnalysis(file);
    }
    throw error;
  }
}
