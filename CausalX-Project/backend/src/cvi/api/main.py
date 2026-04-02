from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from pathlib import Path

from src.cvi.api.inference_service import run_full_cvi_pipeline
from src.cvi.api.background_worker import BackgroundWorker
from src.cvi.api.result_reporting import emit_hidden_score_summary
from src.cvi.api.video_preview import (
    cleanup_preview_cache,
    ensure_video_preview,
    preview_path_for_analysis,
)
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


def _safe_upload_path(filename: str) -> str:
    """
    Collapse user-supplied names to a safe, unique path inside UPLOAD_DIR.
    Avoids path traversal and collisions by stripping directories and
    prefixing with a UUID.
    """
    name = Path(filename).name  # drop any path components
    unique = f"{uuid.uuid4()}_{name}"
    return os.path.join(UPLOAD_DIR, unique)

@app.on_event("startup")
def startup_worker():
    # Keep preview storage bounded across restarts, then start the async worker
    # used by the polling-based analysis flow.
    cleanup_preview_cache(force=True)
    worker.start()


@app.on_event("shutdown")
def shutdown_worker():
    worker.stop()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/preview/{analysis_id}")
async def get_video_preview(analysis_id: str):
    preview_path = preview_path_for_analysis(analysis_id)
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(preview_path, media_type="video/mp4", filename=preview_path.name)


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    analysis_id = str(uuid.uuid4())
    log_event(analysis_id, "upload_received", {"filename": file.filename})
    video_path = _safe_upload_path(file.filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate a lightweight preview first so the UI can render a stable
        # playback source while the heavier inference pipeline runs.
        preview_url = ensure_video_preview(analysis_id, video_path)
        log_event(analysis_id, "processing_started")
        pipeline_output = run_full_cvi_pipeline(video_path)
        frame_results = pipeline_output["frames"]
        if preview_url:
            pipeline_output["preview_url"] = preview_url
        # Emit internal diagnostics without changing the public response shape.
        emit_hidden_score_summary(analysis_id, file.filename, pipeline_output)

        label = "FAKE" if pipeline_output.get("video_fake") else "REAL"

        response = {
            "analysis_id": analysis_id,
            "video_name": file.filename,
            "video_fake": label,
            "fake_confidence": pipeline_output.get("fake_confidence"),
            "overall_score": pipeline_output.get("overall_score"),
            "causal_breach_score": pipeline_output.get("causal_breach_score"),
            "scm_enabled": pipeline_output.get("scm_enabled"),
            "decision_source": pipeline_output.get("decision_source"),
            "legacy_fake_ratio": pipeline_output.get("legacy_fake_ratio"),
            "calibrator_score": pipeline_output.get("calibrator_score"),
            "preview_url": preview_url,
            "highlight_timestamps": pipeline_output.get("highlight_timestamps", []),
            "causal_segments": pipeline_output.get("causal_segments", []),
            "frames": frame_results
        }
        save_result(analysis_id=analysis_id, video_name=file.filename, payload=response)
        log_event(analysis_id, "processing_completed")
        return response
    finally:
        # Synchronous requests are fully resolved in-process, so the uploaded
        # file can be deleted immediately after the response is assembled.
        if os.path.exists(video_path):
            os.remove(video_path)


@app.post("/analyze/async")
async def analyze_video_async(file: UploadFile = File(...)):
    analysis_id = str(uuid.uuid4())
    video_path = _safe_upload_path(file.filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    # The async path keeps the uploaded file on disk because the background
    # worker needs to pick it up after this request returns.
    preview_url = ensure_video_preview(analysis_id, video_path)
    job_id = worker.submit(video_path, job_id=analysis_id)

    return {
        "job_id": job_id,
        "analysis_id": job_id,
        "status": "queued",
        "preview_url": preview_url,
    }


@app.get("/analyze/status/{job_id}")
async def get_job_status(job_id: str, include_result: bool = False):
    record = worker.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": record.job_id,
        "status": record.status,
        # Large frame payloads are omitted unless the caller explicitly asks.
        "result": record.result if include_result else None,
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
