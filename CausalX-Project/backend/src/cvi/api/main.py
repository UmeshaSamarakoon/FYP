from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from src.cvi.api.inference_service import run_full_cvi_pipeline
from src.cvi.api.background_worker import BackgroundWorker

app = FastAPI()
worker = BackgroundWorker()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def startup_worker():
    worker.start()


@app.on_event("shutdown")
def shutdown_worker():
    worker.stop()


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


@app.post("/analyze/async")
async def analyze_video_async(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = worker.submit(video_path)

    return {
        "job_id": job_id,
        "status": "queued"
    }


@app.get("/analyze/status/{job_id}")
async def get_job_status(job_id: str):
    record = worker.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": record.job_id,
        "status": record.status,
        "result": record.result,
        "error": record.error
    }
