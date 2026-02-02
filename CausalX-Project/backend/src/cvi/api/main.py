from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from src.cvi.api.inference_service import run_full_cvi_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    pipeline_output = run_full_cvi_pipeline(video_path)
    frame_results = pipeline_output["frames"]

    label = "FAKE" if pipeline_output.get("video_fake") else "REAL"

    return {
        "video_name": file.filename,
        "video_fake": label,
        "fake_confidence": pipeline_output.get("fake_confidence"),
        "overall_score": pipeline_output.get("overall_score"),
        "highlight_timestamps": pipeline_output.get("highlight_timestamps", []),
        "causal_segments": pipeline_output.get("causal_segments", []),
        "frames": frame_results
    }
