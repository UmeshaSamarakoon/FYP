import os
import torch
import numpy as np
from pathlib import Path
import joblib
import librosa
import warnings
from src.cvi.face_bbox_highlighter import detect_face_bbox, mouth_bbox_from_landmarks
from src.modules.causal_fusion import CausalFusionNetwork, CausalFusionNetworkV2
from src.cvi.feature_extractor import FeatureExtractor
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
EMB_MODEL_PATH = Path(
    os.getenv("CFN_EMB_MODEL_PATH", str(_MODULE_DIR / "models" / "cfn_emb.pth"))
)
DEVICE = torch.device("cpu")

USE_EMBEDDINGS = os.getenv("CFN_USE_EMBEDDINGS", "false").lower() == "true"

if USE_EMBEDDINGS:
    model = CausalFusionNetworkV2(av_dim=4, phys_dim=2).to(DEVICE)
    model.load_state_dict(torch.load(EMB_MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
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

def _load_audio_segment(video_path, offset, duration):
    try:
        waveform, sr = librosa.load(video_path, sr=16000, offset=offset, duration=duration)
        return waveform, sr
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Audio load failed for embeddings: {exc}")
        return np.array([], dtype=np.float32), 16000


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

    feature_extractor = FeatureExtractor()

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

        visual_embedding_scalar = 0.0
        audio_embedding_scalar = 0.0
        if USE_EMBEDDINGS:
            lip_signal = np.array([f["lip_aperture"] for f in frames], dtype=np.float32)
            visual_embedding = feature_extractor.get_visual_embeddings(lip_signal)
            visual_embedding_scalar = float(np.mean(visual_embedding))

            waveform, sr = _load_audio_segment(video_path, offset=chunk_start, duration=current_chunk)
            if waveform.size:
                audio_embedding = feature_extractor.get_audio_embeddings(waveform, sr)
                audio_embedding_scalar = float(np.mean(audio_embedding))

        for i, frame in enumerate(frames):
            if USE_EMBEDDINGS:
                av_features = np.array([
                    frame["lip_aperture"],
                    av_mismatch[i],
                    visual_embedding_scalar,
                    audio_embedding_scalar
                ], dtype=np.float32)
            else:
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
