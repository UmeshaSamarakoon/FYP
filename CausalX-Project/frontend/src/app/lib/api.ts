const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);

export type FrameResult = {
  timestamp: number;
  fake_prob: number;
  av_mismatch?: number;
  bbox?: [number, number, number, number] | null;
};

export type AnalyzeResponse = {
  video_fake: string | number;
  fake_confidence?: number;
  highlight_timestamps?: number[];
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
  const { pollIntervalMs = 2000, timeoutMs = 5 * 60 * 1000 } = options;

  try {
    const submit = await submitAsyncAnalysis(file);
    const start = Date.now();

    while (true) {
      if (Date.now() - start > timeoutMs) {
        throw new Error("Analysis timed out. Please try a shorter video.");
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
