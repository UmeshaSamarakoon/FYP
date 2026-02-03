export async function analyzeVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  const apiBase = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000")
    .replace(/\/+$/, "");

  const response = await fetch(`${apiBase}/analyze`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error("Backend analysis failed");
  }

  return await response.json();
}
