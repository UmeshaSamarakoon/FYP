import os
import numpy as np
from src.cvi.cfn_frame_inference import run_cfn_on_video

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

def run_full_cvi_pipeline(video_path):
    frame_results = run_cfn_on_video(
        video_path,
        threshold=PROB_THRESH,
        causal_threshold=CAUSAL_THRESH,
        chunk_seconds=CHUNK_SECONDS,
        max_seconds=MAX_SECONDS
    )

    # Apply smoothing to reduce false spikes; fallback to raw if window <= 1
    frame_results, prob_key = smooth_fake_probs(frame_results, SMOOTH_WINDOW)

    frame_results = add_causal_breaks(frame_results, causal_thresh=CAUSAL_THRESH)
    causal_segments = build_segments(frame_results, flag_key="causal_break")

    video_fake, confidence, highlight_times = summarize_video(
        frame_results,
        prob_thresh=PROB_THRESH,
        ratio_thresh=RATIO_THRESH,
        prob_key=prob_key
    )

    overall_score = overall_video_score(frame_results, prob_key=prob_key)

    return {
        "video_name": os.path.basename(video_path),
        "video_fake": video_fake,
        "fake_confidence": confidence,
        "overall_score": overall_score,
        "highlight_timestamps": highlight_times,
        "causal_segments": causal_segments,
        "frames": frame_results
    }
