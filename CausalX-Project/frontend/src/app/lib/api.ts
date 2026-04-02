const RAW_API_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";
const API_URL = String(RAW_API_URL).replace(/\/+$/, "");
const DEFAULT_POLL_INTERVAL_MS = Number(import.meta.env.VITE_ANALYSIS_POLL_MS || 2000);
const DEFAULT_TIMEOUT_MS = Number(import.meta.env.VITE_ANALYSIS_TIMEOUT_MS || 15 * 60 * 1000);
const DEFAULT_REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_API_REQUEST_TIMEOUT_MS || 30 * 1000);
const DEFAULT_UPLOAD_TIMEOUT_MS = Number(import.meta.env.VITE_API_UPLOAD_TIMEOUT_MS || 10 * 60 * 1000);
const DEFAULT_RETRY_ATTEMPTS = Number(import.meta.env.VITE_API_RETRY_ATTEMPTS || 3);
const DEFAULT_RETRY_BASE_MS = Number(import.meta.env.VITE_API_RETRY_BASE_MS || 1200);
const DEFAULT_STATUS_ERROR_TOLERANCE = Number(
  import.meta.env.VITE_ANALYSIS_STATUS_ERROR_TOLERANCE || 6,
);
const ALLOW_DIRECT_FALLBACK =
  String(import.meta.env.VITE_ALLOW_DIRECT_FALLBACK || "true").toLowerCase() === "true";

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
  video_fake: string | number | boolean;
  fake_confidence?: number;
  overall_score?: number;
  causal_breach_score?: number;
  scm_enabled?: boolean;
  decision_source?: string;
  legacy_fake_ratio?: number;
  calibrator_score?: number | null;
  preview_url?: string | null;
  highlight_timestamps?: number[];
  causal_segments?: { start: number; end: number; score?: number }[];
  frames: FrameResult[];
};

type AnalyzeAsyncSubmitResponse = {
  job_id: string;
  status: "queued" | "running";
  preview_url?: string | null;
};

type AnalyzeJobStatusResponse = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  result?: AnalyzeResponse;
  error?: string | null;
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

class ApiError extends Error {
  status: number | null;
  body: string;

  constructor(message: string, status: number | null = null, body = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function asPositiveNumber(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function retryDelay(baseMs: number, attempt: number): number {
  const jitterMs = Math.floor(Math.random() * 250);
  return baseMs * Math.max(1, 2 ** attempt) + jitterMs;
}

function toApiUrl(path: string | null | undefined): string | null {
  const value = String(path || "").trim();
  if (!value) return null;
  if (/^https?:\/\//i.test(value)) return value;
  return `${API_URL}${value.startsWith("/") ? value : `/${value}`}`;
}

function isTransientStatus(status: number | null): boolean {
  if (status == null) return true;
  return [408, 425, 429, 500, 502, 503, 504].includes(status);
}

function normalizeApiError(error: unknown, fallbackMessage: string): ApiError {
  if (error instanceof ApiError) return error;

  const anyError = error as any;
  const message = String(anyError?.message || fallbackMessage);
  if (anyError?.name === "AbortError") {
    return new ApiError("Request timed out. The backend took too long to respond.");
  }

  return new ApiError(message || fallbackMessage);
}

async function readErrorBody(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  if (!text) return "";

  try {
    const parsed = JSON.parse(text);
    if (typeof parsed?.detail === "string") return parsed.detail;
  } catch {
    // Ignore parse failures and keep raw text.
  }

  return text;
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal,
    });
  } finally {
    window.clearTimeout(timer);
  }
}

type JsonRequestOptions = {
  timeoutMs?: number;
  retries?: number;
  retryBaseMs?: number;
  label?: string;
};

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  options: JsonRequestOptions = {},
): Promise<T> {
  const timeoutMs = asPositiveNumber(options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS, DEFAULT_REQUEST_TIMEOUT_MS);
  const retries = Math.max(0, Math.floor(asPositiveNumber(options.retries ?? DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_ATTEMPTS)));
  const retryBaseMs = asPositiveNumber(options.retryBaseMs ?? DEFAULT_RETRY_BASE_MS, DEFAULT_RETRY_BASE_MS);
  const label = options.label || "API request";

  for (let attempt = 0; ; attempt += 1) {
    try {
      const res = await fetchWithTimeout(`${API_URL}${path}`, init, timeoutMs);

      if (!res.ok) {
        const body = await readErrorBody(res);
        throw new ApiError(
          body || `${label} failed with status ${res.status}.`,
          res.status,
          body,
        );
      }

      return (await res.json()) as T;
    } catch (error: unknown) {
      const apiError = normalizeApiError(error, `${label} failed.`);
      const isLastAttempt = attempt >= retries;
      if (isLastAttempt || !isTransientStatus(apiError.status)) {
        throw apiError;
      }
      // Retry only transient transport/backend failures with backoff so short
      // outages do not immediately surface as user-facing errors.
      await sleep(retryDelay(retryBaseMs, attempt));
    }
  }
}

function toUserFacingError(error: unknown): Error {
  const apiError = normalizeApiError(error, "Analysis failed.");

  if (apiError.status === 502 || apiError.status === 503 || apiError.status === 504) {
    return new Error(
      `Backend gateway/server is temporarily unavailable (${apiError.status}). Please retry in a few seconds.`,
    );
  }

  if (apiError.status == null && apiError.message.toLowerCase().includes("failed to fetch")) {
    return new Error(
      `Cannot reach backend at ${API_URL}. Make sure the backend server is running and API URL is correct.`,
    );
  }

  return new Error(apiError.message || "Analysis failed.");
}

async function runDirectAnalysis(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<AnalyzeResponse>("/analyze", {
    method: "POST",
    body: formData,
  }, {
    timeoutMs: DEFAULT_UPLOAD_TIMEOUT_MS,
    label: "Direct analysis",
  });
}

async function submitAsyncAnalysis(file: File): Promise<AnalyzeAsyncSubmitResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<AnalyzeAsyncSubmitResponse>("/analyze/async", {
    method: "POST",
    body: formData,
  }, {
    timeoutMs: DEFAULT_UPLOAD_TIMEOUT_MS,
    label: "Async submission",
  });
}

async function fetchAsyncStatus(jobId: string): Promise<AnalyzeJobStatusResponse> {
  return requestJson<AnalyzeJobStatusResponse>(`/analyze/status/${encodeURIComponent(jobId)}`, {
    method: "GET",
  }, {
    timeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
    // Poll loop already retries; avoid multiplying wait per iteration.
    retries: 0,
    label: "Async status check",
  });
}

async function fetchAnalysisResult(jobId: string): Promise<AnalyzeResponse> {
  return requestJson<AnalyzeResponse>(`/results/${encodeURIComponent(jobId)}`, {
    method: "GET",
  }, {
    label: "Result fetch",
  });
}

async function pingBackend(): Promise<void> {
  try {
    await requestJson<{ status: string }>("/health", { method: "GET" }, {
      timeoutMs: 5000,
      retries: 1,
      label: "Health check",
    });
  } catch (error) {
    const apiError = normalizeApiError(error, "Health check failed.");
    // Older backend builds might not expose /health yet.
    if (apiError.status === 404 || apiError.status === 405) return;
    throw apiError;
  }
}

export async function analyzeVideo(
  file: File,
  options: { pollIntervalMs?: number; timeoutMs?: number; statusErrorTolerance?: number } = {},
): Promise<AnalyzeResponse> {
  const {
    pollIntervalMs = asPositiveNumber(DEFAULT_POLL_INTERVAL_MS, 2000),
    timeoutMs = asPositiveNumber(DEFAULT_TIMEOUT_MS, 15 * 60 * 1000),
    statusErrorTolerance = asPositiveNumber(DEFAULT_STATUS_ERROR_TOLERANCE, 6),
  } = options;

  let submittedJobId: string | null = null;

  try {
    // The preferred path is async submission plus polling because inference can
    // outlive a normal request timeout on CPU-only deployments.
    await pingBackend();
    const submit = await submitAsyncAnalysis(file);
    submittedJobId = submit.job_id;
    const previewUrl = toApiUrl(submit.preview_url);
    const start = Date.now();
    let statusErrorCount = 0;

    while (true) {
      if (Date.now() - start > timeoutMs) {
        throw new ApiError(
          `Analysis timed out after ${Math.round(timeoutMs / 1000)}s. Please try a shorter video.`,
        );
      }

      let status: AnalyzeJobStatusResponse;
      try {
        status = await fetchAsyncStatus(submit.job_id);
        statusErrorCount = 0;
      } catch (error) {
        const apiError = normalizeApiError(error, "Async status check failed.");
        // Tolerate a few transient polling failures so the browser does not
        // abandon jobs that are still running correctly on the backend.
        if (isTransientStatus(apiError.status) && statusErrorCount < statusErrorTolerance) {
          statusErrorCount += 1;
          await sleep(pollIntervalMs);
          continue;
        }
        throw apiError;
      }

      if (status.status === "completed") {
        if (status.result) {
          return {
            ...status.result,
            preview_url: toApiUrl(status.result.preview_url) ?? previewUrl,
          };
        }
        const result = await fetchAnalysisResult(submit.job_id);
        return {
          ...result,
          preview_url: toApiUrl(result.preview_url) ?? previewUrl,
        };
      }

      if (status.status === "failed") {
        throw new ApiError(status.error || "Analysis failed on the server.");
      }

      await sleep(pollIntervalMs);
    }
  } catch (error: unknown) {
    const apiError = normalizeApiError(error, "Analysis failed.");

    if (
      ALLOW_DIRECT_FALLBACK &&
      !submittedJobId &&
      (apiError.status === 404 || apiError.status === 405)
    ) {
      // Older backends may not have the async endpoints yet, so fall back to
      // the original blocking `/analyze` route before surfacing an error.
      try {
        const result = await runDirectAnalysis(file);
        return {
          ...result,
          preview_url: toApiUrl(result.preview_url),
        };
      } catch (fallbackError) {
        throw toUserFacingError(fallbackError);
      }
    }

    throw toUserFacingError(apiError);
  }
}
