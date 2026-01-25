export async function analyzeVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://127.0.0.1:8000/analyze", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    throw new Error("Backend analysis failed");
  }

  return await response.json();
}
