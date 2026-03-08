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
    compute_av_sync_signals,
    get_video_meta
)
from src.cvi.feature_schema import (
    BASELINE_AV_FEATURES,
    EXTENDED_AV_FEATURES,
    BASELINE_PHYS_FEATURES,
    EXTENDED_PHYS_FEATURES,
)

# --------------------------------------------------
# Load trained CFN model
# --------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = _MODULE_DIR / "models" / "cfn.pth"
DEVICE = torch.device("cpu")

_model = None
_scaler = None
_AV_DIM = None
_PHYS_DIM = None
_models = []
_scalers = []
_AV_DIMS = []
_PHYS_DIMS = []
_scaler_shape_warned = set()
_AV_FEATURE_ORDER = list(BASELINE_AV_FEATURES) + list(EXTENDED_AV_FEATURES) + ["tcn_visual_emb", "wav2vec_audio_emb"]
_PHYS_FEATURE_ORDER = list(BASELINE_PHYS_FEATURES) + list(EXTENDED_PHYS_FEATURES)


def _use_embeddings() -> bool:
    return os.getenv("CFN_USE_EMBEDDINGS", "false").lower() == "true"


def _split_env_paths(value: str):
    return [Path(p.strip()) for p in value.split(",") if p.strip()]


def _build_model_from_state(state, use_emb):
    av_dim = int(state.get("av_branch.0.weight", torch.empty(0)).shape[1]) if state else 0
    phys_dim = int(state.get("physical_branch.0.weight", torch.empty(0)).shape[1]) if state else 0
    if av_dim <= 0:
        av_dim = 4 if use_emb else 3
    if phys_dim <= 0:
        phys_dim = 2

    # Keep backward compatibility with legacy fixed-dim checkpoints.
    if not use_emb and av_dim == 3 and phys_dim == 2:
        model = CausalFusionNetwork().to(DEVICE)
    else:
        model = CausalFusionNetworkV2(av_dim=av_dim, phys_dim=phys_dim).to(DEVICE)
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        warnings.warn(
            "CFN checkpoint loaded with missing keys (using default init): "
            + ", ".join(load_result.missing_keys)
        )
    if load_result.unexpected_keys:
        warnings.warn(
            "CFN checkpoint has unexpected keys (ignored): "
            + ", ".join(load_result.unexpected_keys)
        )
    model.eval()
    return model, av_dim, phys_dim


def _load_artifacts():
    """
    Lazy-load model and scaler so API can start even if artifacts are absent.
    Raises a clear RuntimeError when files are missing or corrupted.
    """
    global _model, _scaler, _AV_DIM, _PHYS_DIM, _models, _scalers, _AV_DIMS, _PHYS_DIMS
    if _models:
        return

    try:
        use_emb = _use_embeddings()
        ensemble_model_paths_raw = os.getenv("CFN_ENSEMBLE_MODEL_PATHS", "").strip()
        ensemble_scaler_paths_raw = os.getenv("CFN_ENSEMBLE_SCALER_PATHS", "").strip()

        if ensemble_model_paths_raw:
            model_paths = _split_env_paths(ensemble_model_paths_raw)
            if not model_paths:
                raise RuntimeError("CFN_ENSEMBLE_MODEL_PATHS is set but no valid paths found.")

            if ensemble_scaler_paths_raw:
                scaler_paths = _split_env_paths(ensemble_scaler_paths_raw)
                if len(scaler_paths) != len(model_paths):
                    raise RuntimeError("CFN_ENSEMBLE_SCALER_PATHS length must match CFN_ENSEMBLE_MODEL_PATHS.")
            else:
                scaler_paths = []
                for mp in model_paths:
                    # Infer sibling scaler path for matrix artifacts, fallback to default scaler.
                    name = mp.name
                    if name.startswith("cfn_emb_"):
                        run_id = name.replace("cfn_emb_", "").replace(".pth", "")
                        inferred = mp.parent / f"cfn_scaler_{run_id}.pkl"
                        scaler_paths.append(inferred if inferred.exists() else (_MODULE_DIR / "models" / "cfn_scaler.pkl"))
                    else:
                        scaler_paths.append(_MODULE_DIR / "models" / "cfn_scaler.pkl")

            for model_path, scaler_path in zip(model_paths, scaler_paths):
                state = torch.load(model_path, map_location=DEVICE)
                model, av_dim, phys_dim = _build_model_from_state(state, use_emb=use_emb)
                _models.append(model)
                _AV_DIMS.append(av_dim)
                _PHYS_DIMS.append(phys_dim)
                if scaler_path.exists():
                    _scalers.append(joblib.load(scaler_path))
                else:
                    _scalers.append(None)
        else:
            emb_model_path = Path(os.getenv("CFN_EMB_MODEL_PATH", _MODULE_DIR / "models" / "cfn_emb.pth"))
            model_path = emb_model_path if use_emb else MODEL_PATH
            state = torch.load(model_path, map_location=DEVICE)
            model, av_dim, phys_dim = _build_model_from_state(state, use_emb=use_emb)
            _models = [model]
            _AV_DIMS = [av_dim]
            _PHYS_DIMS = [phys_dim]

            scaler_path = _MODULE_DIR / "models" / "cfn_scaler.pkl"
            if scaler_path.exists():
                _scalers = [joblib.load(scaler_path)]
            else:
                _scalers = [None]

        # Backward-compatible single-model aliases
        _model = _models[0]
        _AV_DIM = _AV_DIMS[0]
        _PHYS_DIM = _PHYS_DIMS[0]
        _scaler = _scalers[0]
    except FileNotFoundError as exc:  # noqa: PERF203
        raise RuntimeError(
            "CFN model weights not found. Set CFN_EMB_MODEL_PATH / models/cfn.pth."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load CFN model: {exc}") from exc

    if len(_scalers) != len(_models):
        raise RuntimeError("Scaler/model count mismatch in CFN artifacts.")
    if len(_AV_DIMS) != len(_models) or len(_PHYS_DIMS) != len(_models):
        raise RuntimeError("Feature dimension metadata mismatch in CFN artifacts.")


def _prepare_features_for_model(base_av_vals, base_phys_vals, av_dim, phys_dim, scaler):
    av_vals = list(base_av_vals)
    if av_dim and len(av_vals) < av_dim:
        av_vals.extend([0.0] * (av_dim - len(av_vals)))
    if av_dim and len(av_vals) > av_dim:
        av_vals = av_vals[:av_dim]

    phys_vals = list(base_phys_vals)
    if phys_dim and len(phys_vals) < phys_dim:
        phys_vals.extend([0.0] * (phys_dim - len(phys_vals)))
    if phys_dim and len(phys_vals) > phys_dim:
        phys_vals = phys_vals[:phys_dim]

    av_features = np.array(av_vals, dtype=np.float32)
    phys_features = np.array(phys_vals, dtype=np.float32)

    if scaler is not None:
        try:
            av_features = scaler["av"].transform([av_features])[0].astype(np.float32, copy=False)
            phys_features = scaler["phys"].transform([phys_features])[0].astype(np.float32, copy=False)
        except ValueError as exc:
            shape_key = (int(av_dim or -1), int(phys_dim or -1), str(exc))
            if shape_key not in _scaler_shape_warned:
                warnings.warn(f"Skipping scaler due to shape mismatch: {exc}")
                _scaler_shape_warned.add(shape_key)
            av_features = av_features.astype(np.float32, copy=False)
            phys_features = phys_features.astype(np.float32, copy=False)

    return av_features, phys_features


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


def _ordered_feature_values(feature_map, ordered_cols):
    return [float(feature_map.get(c, 0.0)) for c in ordered_cols]


def run_cfn_on_video(
    video_path,
    threshold=0.6,
    causal_threshold=None,
    chunk_seconds=10,
    max_seconds=None,
    target_fps=None,
    include_bboxes=True,
):
    """
    Returns per-frame CFN predictions with timestamps and bounding boxes.
    Processes the video in chunks to keep memory bounded and only draws
    bboxes on frames flagged as fake.
    """

    _load_artifacts()

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
    use_emb = _use_embeddings()

    while chunk_start < total_duration:
        current_chunk = min(chunk_seconds, total_duration - chunk_start)

        frames = extract_frame_level_features(
            video_path,
            start_time=chunk_start,
            duration=current_chunk,
            fps=fps,
            target_fps=target_fps,
            include_frame=include_bboxes,
            include_landmarks=include_bboxes,
        )

        if len(frames) == 0:
            chunk_start += current_chunk
            continue

        sync_main = compute_av_sync_signals(frames, window=5)
        sync_w3 = compute_av_sync_signals(frames, window=3)
        sync_w6 = compute_av_sync_signals(frames, window=6)
        sync_w12 = compute_av_sync_signals(frames, window=12)
        av_mismatch = sync_main["mismatch"]

        ts = np.array([f.get("timestamp", 0.0) for f in frames], dtype=np.float32)
        if len(ts) >= 2:
            dt = float(np.median(np.diff(ts)))
            effective_fps = 1.0 / max(dt, 1e-6)
        else:
            effective_fps = float(target_fps or fps or 30.0)

        visual_embedding_scalar = 0.0
        audio_embedding_scalar = 0.0
        if use_emb:
            lip_signal = np.array([f["lip_aperture"] for f in frames], dtype=np.float32)
            visual_embedding = feature_extractor.get_visual_embeddings(lip_signal)
            visual_embedding_scalar = float(np.mean(visual_embedding)) if visual_embedding.size else 0.0

            waveform, sr = _load_audio_segment(video_path, offset=chunk_start, duration=current_chunk)
            if waveform.size:
                audio_embedding = feature_extractor.get_audio_embeddings(waveform, sr)
                audio_embedding_scalar = float(np.mean(audio_embedding)) if audio_embedding.size else 0.0

        for i, frame in enumerate(frames):
            # Build richer AV/physical feature maps; model-specific prep truncates/pads as needed.
            av_map = {
                "lip_variance": abs(float(frame.get("lip_velocity", frame.get("lip_aperture", 0.0)))),
                "av_correlation": float(sync_main["local_corr"][i]),
                "av_lag_frames": float(sync_main["local_lag"][i]),
                "lip_mean": float(frame.get("lip_aperture", 0.0)),
                "lip_std": abs(float(frame.get("lip_velocity", 0.0))),
                "lip_range": abs(float(frame.get("lip_velocity", 0.0))),
                "lip_velocity_mean": float(frame.get("lip_velocity", 0.0)),
                "lip_velocity_std": abs(float(frame.get("lip_velocity", 0.0))),
                "audio_rms_mean": float(frame.get("audio_rms", 0.0)),
                "audio_rms_std": abs(float(frame.get("audio_delta", 0.0))),
                "av_corr_05_mean": float(sync_w3["local_corr"][i]),
                "av_corr_05_std": float(sync_w3["local_corr_std"][i]),
                "av_corr_10_mean": float(sync_w6["local_corr"][i]),
                "av_corr_10_std": float(sync_w6["local_corr_std"][i]),
                "av_corr_20_mean": float(sync_w12["local_corr"][i]),
                "av_corr_20_std": float(sync_w12["local_corr_std"][i]),
                "av_peak_corr": float(sync_main["peak_corr"][i]),
                "av_peak_lag_sec": float(sync_main["local_lag"][i]) / max(float(effective_fps), 1e-6),
                "av_peak_prominence": float(sync_main["peak_prominence"][i]),
                "av_onset_corr": float(sync_main["onset_corr"][i]),
                "tcn_visual_emb": float(visual_embedding_scalar if use_emb else 0.0),
                "wav2vec_audio_emb": float(audio_embedding_scalar if use_emb else 0.0),
            }

            phys_map = {
                "jitter_mean": float(frame.get("jitter", 0.0)),
                "jitter_std": float(frame.get("jitter_std", 0.0)),
                "mouth_flow_mean": float(frame.get("mouth_motion", frame.get("mouth_area_delta", 0.0))),
                "mouth_flow_std": float(frame.get("mouth_motion_std", frame.get("mouth_area_delta", 0.0))),
                "mouth_aspect_mean": float(frame.get("mouth_aspect", 0.0)),
                "mouth_aspect_std": float(frame.get("mouth_aspect_delta", 0.0)),
                "mouth_area_mean": float(frame.get("mouth_area_norm", 0.0)),
                "mouth_area_std": float(frame.get("mouth_area_delta", 0.0)),
                "mouth_area_delta_std": float(frame.get("mouth_area_delta", 0.0)),
                "mouth_asym_mean": float(frame.get("mouth_asym", 0.0)),
                "mouth_asym_std": float(frame.get("mouth_asym_delta", 0.0)),
                "det_count": 1.0,
            }

            base_av_vals = _ordered_feature_values(av_map, _AV_FEATURE_ORDER)
            base_phys_vals = _ordered_feature_values(phys_map, _PHYS_FEATURE_ORDER)

            probs = []
            with torch.no_grad():
                for model, scaler, av_dim, phys_dim in zip(_models, _scalers, _AV_DIMS, _PHYS_DIMS):
                    av_features, phys_features = _prepare_features_for_model(
                        base_av_vals, base_phys_vals, av_dim, phys_dim, scaler
                    )
                    X_av = torch.tensor(av_features).unsqueeze(0).to(DEVICE)
                    X_phys = torch.tensor(phys_features).unsqueeze(0).to(DEVICE)
                    probs.append(model(X_av, X_phys).item())
            prob = float(np.mean(probs)) if probs else 0.0

            bbox = None
            if include_bboxes and (prob >= threshold or (causal_threshold is not None and av_mismatch[i] >= causal_threshold)):
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
