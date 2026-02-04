from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from src.cvi.api.inference_service import run_full_cvi_pipeline
from src.cvi.api.background_worker import BackgroundWorker
from src.cvi.storage.results_store import get_result, list_results, save_result
from src.cvi.storage.logs_store import list_logs, log_event

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
    analysis_id = str(uuid.uuid4())
    log_event(analysis_id, "upload_received", {"filename": file.filename})
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    log_event(analysis_id, "processing_started")
    pipeline_output = run_full_cvi_pipeline(video_path)
    frame_results = pipeline_output["frames"]

    label = "FAKE" if pipeline_output.get("video_fake") else "REAL"

    response = {
        "analysis_id": analysis_id,
        "video_name": file.filename,
        "video_fake": label,
        "fake_confidence": pipeline_output.get("fake_confidence"),
        "overall_score": pipeline_output.get("overall_score"),
        "highlight_timestamps": pipeline_output.get("highlight_timestamps", []),
        "causal_segments": pipeline_output.get("causal_segments", []),
        "frames": frame_results
    }
    save_result(analysis_id=analysis_id, video_name=file.filename, payload=response)
    log_event(analysis_id, "processing_completed")
    return response


@app.post("/analyze/async")
async def analyze_video_async(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = worker.submit(video_path)

    return {
        "job_id": job_id,
        "analysis_id": job_id,
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


@app.get("/results")
async def list_analysis_results(limit: int = 50):
    records = list_results(limit=limit)
    return [
        {
            "analysis_id": r.analysis_id,
            "video_name": r.video_name,
            "created_at": r.created_at,
        }
        for r in records
    ]


@app.get("/results/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    record = get_result(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Result not found")
    return record.payload


@app.get("/logs")
async def get_logs(analysis_id: str | None = None, limit: int = 200):
    records = list_logs(analysis_id=analysis_id, limit=limit)
    return [
        {
            "log_id": r.log_id,
            "analysis_id": r.analysis_id,
            "event": r.event,
            "created_at": r.created_at,
            "metadata": r.metadata,
        }
        for r in records
    ]
