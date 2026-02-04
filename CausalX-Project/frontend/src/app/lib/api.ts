const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

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

export async function analyzeVideo(file: File): Promise<AnalyzeResponse> {
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
