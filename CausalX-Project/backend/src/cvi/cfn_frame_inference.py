import torch
import numpy as np
from pathlib import Path
import joblib
from src.cvi.face_bbox_highlighter import detect_face_bbox, mouth_bbox_from_landmarks
from src.modules.causal_fusion import CausalFusionNetwork
from src.cvi.frame_causal_extractor import (
    extract_frame_level_features,
    compute_av_mismatch,
    get_video_meta
)

# --------------------------------------------------
# Load trained CFN model
# --------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = _MODULE_DIR / "models" / "cfn.pth"
DEVICE = torch.device("cpu")

model = CausalFusionNetwork().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

SCALER_PATH = _MODULE_DIR / "models" / "cfn_scaler.pkl"
_scaler = None
if SCALER_PATH.exists():
    _scaler = joblib.load(SCALER_PATH)


# --------------------------------------------------
# Frame-level CFN inference
# --------------------------------------------------

def run_cfn_on_video(
    video_path,
    threshold=0.6,
    causal_threshold=None,
    chunk_seconds=10,
    max_seconds=None
):
    """
    Returns per-frame CFN predictions with timestamps and bounding boxes.
    Processes the video in chunks to keep memory bounded and only draws
    bboxes on frames flagged as fake.
    """

    fps, duration = get_video_meta(video_path)
    if max_seconds is not None:
        total_duration = min(duration, max_seconds) if duration > 0 else max_seconds
    else:
        total_duration = duration

    # If duration metadata is missing, process at least one chunk
    if total_duration <= 0:
        total_duration = chunk_seconds

    results = []
    chunk_start = 0.0

    while chunk_start < total_duration:
        current_chunk = min(chunk_seconds, total_duration - chunk_start)

        frames = extract_frame_level_features(
            video_path,
            start_time=chunk_start,
            duration=current_chunk,
            fps=fps
        )

        if len(frames) == 0:
            chunk_start += current_chunk
            continue

        av_mismatch = compute_av_mismatch(frames)

        for i, frame in enumerate(frames):
            av_features = np.array([
                frame["lip_aperture"],
                av_mismatch[i],
                0.0
            ], dtype=np.float32)

            phys_features = np.array([
                frame.get("jitter", 0.0),
                frame.get("jitter_std", 0.0)
            ], dtype=np.float32)

            if _scaler is not None:
                av_features = _scaler["av"].transform([av_features])[0]
                phys_features = _scaler["phys"].transform([phys_features])[0]

            X_av = torch.tensor(av_features).unsqueeze(0).to(DEVICE)
            X_phys = torch.tensor(phys_features).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                prob = model(X_av, X_phys).item()

            bbox = None
            if prob >= threshold or (causal_threshold is not None and av_mismatch[i] >= causal_threshold):
                bbox = mouth_bbox_from_landmarks(
                    frame.get("landmarks"),
                    frame["frame"].shape
                )
                if bbox is None:
                    bbox = detect_face_bbox(frame["frame"])

            results.append({
                "timestamp": frame["timestamp"],
                "fake_prob": float(prob),
                "av_mismatch": float(av_mismatch[i]),
                "bbox": bbox
            })

        chunk_start += current_chunk

    return results
