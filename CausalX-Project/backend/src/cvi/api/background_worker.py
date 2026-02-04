import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from src.cvi.api.inference_service import run_full_cvi_pipeline


@dataclass
class JobRecord:
    job_id: str
    video_path: str
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    result: dict | None = None
    error: str | None = None


class BackgroundWorker:
    def __init__(self):
        self._queue: queue.Queue[JobRecord] = queue.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def submit(self, video_path: str) -> str:
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id, video_path=video_path)
        self._jobs[job_id] = record
        self._queue.put(record)
        return job_id

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                record = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            record.status = "running"
            try:
                record.result = run_full_cvi_pipeline(record.video_path)
                record.status = "completed"
            except Exception as exc:  # noqa: BLE001
                record.status = "failed"
                record.error = str(exc)
            finally:
                self._queue.task_done()
                if os.path.exists(record.video_path):
                    os.remove(record.video_path)
