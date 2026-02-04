import os
import numpy as np
from src.cvi.pipeline import CausalInferenceEngine, InferenceController

PROB_THRESH = float(os.getenv("CFN_PROB_THRESH", "0.6"))
RATIO_THRESH = float(os.getenv("CFN_RATIO_THRESH", "0.3"))
SMOOTH_WINDOW = int(os.getenv("CFN_SMOOTH_WINDOW", "5"))
CHUNK_SECONDS = int(os.getenv("CFN_CHUNK_SECONDS", "10"))
CAUSAL_THRESH = float(os.getenv("CFN_CAUSAL_THRESH", "0.6"))
MAX_SECONDS_ENV = os.getenv("CFN_MAX_SECONDS")
MAX_SECONDS = float(MAX_SECONDS_ENV) if MAX_SECONDS_ENV else None

def smooth_fake_probs(frames, window):
    """
    Apply simple moving average smoothing over fake_prob.
    """
    if window <= 1 or not frames:
        return frames, "fake_prob"

    probs = np.array([f.get("fake_prob", 0.0) for f in frames], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(probs, kernel, mode="same")

    for f, s in zip(frames, smoothed):
        f["fake_prob_smooth"] = float(s)

    return frames, "fake_prob_smooth"

def summarize_video(frames, prob_thresh=0.6, ratio_thresh=0.3, prob_key="fake_prob"):
    """
    Decide if video is fake based on proportion of suspicious frames
    using the chosen probability key (raw or smoothed).
    """
    if not frames:
        return 0, 0.0, []

    suspicious_frames = [
        f for f in frames if f.get(prob_key, 0.0) >= prob_thresh
    ]

    fake_ratio = len(suspicious_frames) / len(frames)
    video_fake = int(fake_ratio >= ratio_thresh)

    highlight_times = (
        [f["timestamp"] for f in suspicious_frames]
        if video_fake else []
    )

    return video_fake, fake_ratio, highlight_times

def add_causal_breaks(frames, causal_thresh=0.6):
    """
    Tag frames where causal link appears broken based on AV mismatch.
    """
    for f in frames:
        mismatch = f.get("av_mismatch", 0.0)
        f["causal_break"] = bool(mismatch >= causal_thresh)
    return frames

def build_segments(frames, flag_key="causal_break"):
    """
    Build contiguous time segments from frame-level flags.
    """
    flagged = [f for f in frames if f.get(flag_key)]
    if not flagged:
        return []

    timestamps = sorted(f["timestamp"] for f in flagged)
    if len(timestamps) == 1:
        t = timestamps[0]
        return [[t, t]]

    diffs = np.diff(timestamps)
    step = float(np.median(diffs)) if len(diffs) else 0.05
    max_gap = step * 1.5 if step > 0 else 0.1

    segments = []
    start = timestamps[0]
    prev = timestamps[0]

    for t in timestamps[1:]:
        if t - prev > max_gap:
            segments.append([start, prev])
            start = t
        prev = t

    segments.append([start, prev])
    return segments

def overall_video_score(frames, prob_key="fake_prob"):
    if not frames:
        return 0.0
    return float(np.mean([f.get(prob_key, 0.0) for f in frames]))

def build_inference_controller() -> InferenceController:
    engine = CausalInferenceEngine(
        prob_thresh=PROB_THRESH,
        ratio_thresh=RATIO_THRESH,
        smooth_window=SMOOTH_WINDOW,
        chunk_seconds=CHUNK_SECONDS,
        causal_thresh=CAUSAL_THRESH,
        max_seconds=MAX_SECONDS,
    )
    return InferenceController(engine=engine)


def run_full_cvi_pipeline(video_path):
    controller = build_inference_controller()
    output = controller.process(video_path)
    output["video_name"] = os.path.basename(video_path)
    return output
