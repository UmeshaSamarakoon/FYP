import os
import json
import torch
import numpy as np
from pathlib import Path
import joblib
import librosa
import warnings
from src.cvi.face_bbox_highlighter import detect_face_bbox, mouth_bbox_from_landmarks
from src.modules.causal_fusion import CausalFusionNetworkV2
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
    LIP_STREAM_FEATURES,
)

# --------------------------------------------------
# Load trained CFN model
# --------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cpu")

_model = None
_scaler = None
_AV_DIM = None
_PHYS_DIM = None
_LIP_DIM = None
_models = []
_scalers = []
_AV_DIMS = []
_PHYS_DIMS = []
_LIP_DIMS = []
_scaler_shape_warned = set()
_AV_FEATURE_ORDER = list(BASELINE_AV_FEATURES) + list(EXTENDED_AV_FEATURES) + [
    "tcn_visual_emb",
    "wav2vec_audio_emb",
    "effnet_b4_face_emb",
    "lip_roi_emb",
    "wav2vec2_base_ft_emb",
]
_PHYS_FEATURE_ORDER = list(BASELINE_PHYS_FEATURES) + list(EXTENDED_PHYS_FEATURES)
_LIP_FEATURE_ORDER = list(LIP_STREAM_FEATURES)
_DEFAULT_EMB_MODEL_PATH = _MODULE_DIR / "models" / "cfn_emb.pth"
_DEFAULT_ENSEMBLE_MANIFEST_PATH = _MODULE_DIR / "models" / "fakeavceleb_best_step46_multiseed_manifest.json"
_T2A_ENABLE = os.getenv("CFN_T2A_ENABLE", "false").lower() == "true"
_T2A_TARGET_ENTROPY = float(os.getenv("CFN_T2A_TARGET_ENTROPY", "0.58"))
_T2A_MAX_TEMP = float(os.getenv("CFN_T2A_MAX_TEMP", "2.5"))
_T2A_MIN_FRAMES = int(os.getenv("CFN_T2A_MIN_FRAMES", "24"))
_TEMP_SCALE_FALLBACK = float(os.getenv("CFN_TEMP_SCALE", "1.0"))
_TEMP_SCALE_PATH = os.getenv("CFN_TEMP_SCALE_PATH", "").strip() or None
_RUNTIME_TEMP_SCALE = None
_FEATURE_Z_CLIP = float(os.getenv("CFN_FEATURE_Z_CLIP", "5.0"))


def _use_embeddings() -> bool:
    return os.getenv("CFN_USE_EMBEDDINGS", "true").lower() == "true"


def _clip_prob(p):
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def _apply_temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = float(max(temperature, 1e-4))
    p = np.asarray(_clip_prob(probs), dtype=np.float64)
    logits = np.log(p / (1.0 - p)) / t
    out = 1.0 / (1.0 + np.exp(-logits))
    return np.asarray(_clip_prob(out), dtype=np.float32)


def _binary_entropy(p: np.ndarray) -> float:
    x = np.asarray(_clip_prob(p), dtype=np.float64)
    return float(np.mean(-(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))))


def _estimate_t2a_temperature(
    probs: np.ndarray,
    target_entropy: float,
    max_temp: float,
) -> float:
    p = np.asarray(probs, dtype=np.float32)
    if p.size < 2:
        return 1.0
    target = float(np.clip(target_entropy, 0.1, 0.69))
    lo, hi = 1.0, float(max(1.0, max_temp))
    best_t = 1.0
    best_gap = abs(_binary_entropy(p) - target)
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        ent = _binary_entropy(_apply_temperature_scale(p, mid))
        gap = abs(ent - target)
        if gap < best_gap:
            best_gap = gap
            best_t = mid
        if ent < target:
            lo = mid
        else:
            hi = mid
    return float(best_t)


def _resolve_runtime_temperature() -> float:
    global _RUNTIME_TEMP_SCALE
    if _RUNTIME_TEMP_SCALE is not None:
        return float(_RUNTIME_TEMP_SCALE)

    temp = float(_TEMP_SCALE_FALLBACK)
    candidate_paths = []
    if _TEMP_SCALE_PATH:
        candidate_paths.append(Path(_TEMP_SCALE_PATH))
    else:
        explicit_emb_model = os.getenv("CFN_EMB_MODEL_PATH", "").strip()
        explicit_ensemble_paths = os.getenv("CFN_ENSEMBLE_MODEL_PATHS", "").strip()
        manifest_path = _resolve_manifest_path(single_model_override=explicit_emb_model)
        if explicit_emb_model:
            try:
                candidate_paths.append(Path(explicit_emb_model).parent / "cfn_temperature.json")
            except Exception:
                pass
        elif not explicit_ensemble_paths and manifest_path is None:
            try:
                candidate_paths.append(_DEFAULT_EMB_MODEL_PATH.parent / "cfn_temperature.json")
            except Exception:
                pass
        candidate_paths.append(_MODULE_DIR / "models" / "cfn_temperature.json")

    for p in candidate_paths:
        try:
            if p.exists():
                obj = json.loads(p.read_text())
                if isinstance(obj, dict) and "temperature" in obj:
                    temp = float(obj["temperature"])
                    break
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to load CFN temperature file '{p}': {exc}")
    _RUNTIME_TEMP_SCALE = float(max(temp, 1e-4))
    return float(_RUNTIME_TEMP_SCALE)


def _split_env_paths(value: str):
    return [Path(p.strip()) for p in value.split(",") if p.strip()]


def _resolve_relative_path(value, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p)


def _default_manifest_has_all_model_paths() -> bool:
    manifest_path = _DEFAULT_ENSEMBLE_MANIFEST_PATH
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        return False

    entries = payload.get("artifacts")
    if not isinstance(entries, list) or not entries:
        return False

    for entry in entries:
        if not isinstance(entry, dict):
            return False
        model_path = entry.get("model_path")
        if not model_path:
            return False
        if not _resolve_relative_path(model_path, manifest_path.parent).exists():
            return False
    return True


def _resolve_manifest_path(single_model_override: str = ""):
    explicit = os.getenv("CFN_ENSEMBLE_MANIFEST_PATH", "").strip()
    if explicit:
        return Path(explicit)
    return None


def _load_manifest_artifacts(manifest_path: Path):
    payload = json.loads(manifest_path.read_text())
    entries = payload.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"Manifest '{manifest_path}' has no artifact entries.")

    model_paths = []
    scaler_paths = []
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Manifest '{manifest_path}' entry #{idx} is not an object.")
        model_path = entry.get("model_path")
        scaler_path = entry.get("scaler_path")
        if not model_path:
            raise RuntimeError(f"Manifest '{manifest_path}' entry #{idx} is missing model_path.")
        resolved_model = _resolve_relative_path(model_path, manifest_path.parent)
        model_paths.append(resolved_model)
        if scaler_path:
            scaler_paths.append(_resolve_relative_path(scaler_path, manifest_path.parent))
        else:
            scaler_paths.append(_resolve_single_scaler_path(resolved_model))
    return model_paths, scaler_paths


def _resolve_single_scaler_path(model_path: Path) -> Path:
    explicit = os.getenv("CFN_SCALER_PATH", "").strip()
    if explicit:
        return Path(explicit)

    sibling = model_path.parent / "cfn_scaler.pkl"
    if sibling.exists():
        return sibling
    return _MODULE_DIR / "models" / "cfn_scaler.pkl"


def _resolve_model_paths_for_threshold() -> list[Path]:
    """
    Infer the candidate CFN checkpoint locations used for inference so we can read their threshold reports.
    Matches the branching logic from _load_artifacts().
    """
    # Keep this in sync with `_load_artifacts()` so we know exactly which thresholds are relevant.
    ensemble_model_paths_raw = os.getenv("CFN_ENSEMBLE_MODEL_PATHS", "").strip()
    single_model_override = os.getenv("CFN_EMB_MODEL_PATH", "").strip()
    manifest_path = _resolve_manifest_path(single_model_override=single_model_override)

    if ensemble_model_paths_raw:
        paths = _split_env_paths(ensemble_model_paths_raw)
        if not paths:
            raise RuntimeError("CFN_ENSEMBLE_MODEL_PATHS is set but no valid paths were discovered for threshold lookup.")
        return paths
    if manifest_path is not None:
        model_paths, _ = _load_manifest_artifacts(manifest_path)
        return model_paths
    if single_model_override:
        return [Path(single_model_override)]
    return [_DEFAULT_EMB_MODEL_PATH]


def _load_threshold_from_model_dir(model_path: Path) -> float | None:
    report_path = model_path.parent / "cfn_threshold_report.json"
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text())
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Failed to parse CFN threshold file '{report_path}': {exc}")
        return None
    # Read the selected threshold so inference can default to the training decision boundary.
    chosen = payload.get("chosen_epoch_report") or {}
    threshold = chosen.get("selection_threshold")
    if threshold is None:
        return None
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return None


def resolve_default_probability_threshold() -> float | None:
    """
    Return the mean selection threshold recorded next to the checkpoints that will be loaded, if available.
    """
    # Averaging thresholds allows multi-checkpoint ensembles (e.g., manifest entries) to share a single default.
    values = []
    for model_path in _resolve_model_paths_for_threshold():
        thr = _load_threshold_from_model_dir(model_path)
        if thr is not None and np.isfinite(thr):
            values.append(float(thr))
    if not values:
        return None
    return float(np.mean(values))


def _build_model_from_state(state, use_emb):
    def _infer_in_dim(branch_key: str) -> int:
        if not isinstance(state, dict):
            return 0
        w = state.get(branch_key)
        if isinstance(w, torch.Tensor) and w.ndim >= 2:
            return int(w.shape[1])
        return 0

    av_dim = _infer_in_dim("av_branch.0.weight")
    phys_dim = _infer_in_dim("physical_branch.0.weight")
    lip_dim = _infer_in_dim("lip_branch.0.weight")
    enable_causal_breach_head = bool(
        isinstance(state, dict) and any(k.startswith("causal_breach_head.") for k in state.keys())
    )
    enable_av_input_layernorm = bool(
        isinstance(state, dict) and any(k.startswith("av_input_ln.") for k in state.keys())
    )
    if av_dim <= 0:
        av_dim = 4 if use_emb else 3
    if phys_dim <= 0:
        phys_dim = 2
    if lip_dim < 0:
        lip_dim = 0
    if lip_dim > 0:
        raise RuntimeError(
            "CFN lip-branch checkpoints are not supported by the current runtime. "
            "Use the active two-branch V2 ensemble or restore a compatible 3-branch model definition."
        )
    model = CausalFusionNetworkV2(
        av_dim=av_dim,
        phys_dim=phys_dim,
        lip_dim=0,
        enable_causal_breach_head=enable_causal_breach_head,
        enable_av_input_layernorm=enable_av_input_layernorm,
    ).to(DEVICE)
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
    return model, av_dim, phys_dim, 0


def _load_artifacts():
    """
    Lazy-load model and scaler so API can start even if artifacts are absent.
    Raises a clear RuntimeError when files are missing or corrupted.
    """
    global _model, _scaler, _AV_DIM, _PHYS_DIM, _LIP_DIM, _models, _scalers, _AV_DIMS, _PHYS_DIMS, _LIP_DIMS
    if _models:
        return

    try:
        use_emb = _use_embeddings()
        ensemble_model_paths_raw = os.getenv("CFN_ENSEMBLE_MODEL_PATHS", "").strip()
        ensemble_scaler_paths_raw = os.getenv("CFN_ENSEMBLE_SCALER_PATHS", "").strip()
        single_model_override = os.getenv("CFN_EMB_MODEL_PATH", "").strip()
        manifest_path = _resolve_manifest_path(single_model_override=single_model_override)

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
                    sibling = mp.parent / "cfn_scaler.pkl"
                    if sibling.exists():
                        scaler_paths.append(sibling)
                        continue
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
                model, av_dim, phys_dim, lip_dim = _build_model_from_state(state, use_emb=use_emb)
                _models.append(model)
                _AV_DIMS.append(av_dim)
                _PHYS_DIMS.append(phys_dim)
                _LIP_DIMS.append(lip_dim)
                if scaler_path.exists():
                    _scalers.append(joblib.load(scaler_path))
                else:
                    _scalers.append(None)
        elif manifest_path is not None:
            model_paths, scaler_paths = _load_manifest_artifacts(manifest_path)
            for model_path, scaler_path in zip(model_paths, scaler_paths):
                state = torch.load(model_path, map_location=DEVICE)
                model, av_dim, phys_dim, lip_dim = _build_model_from_state(state, use_emb=use_emb)
                _models.append(model)
                _AV_DIMS.append(av_dim)
                _PHYS_DIMS.append(phys_dim)
                _LIP_DIMS.append(lip_dim)
                if scaler_path.exists():
                    _scalers.append(joblib.load(scaler_path))
                else:
                    _scalers.append(None)
        else:
            model_path = Path(single_model_override or _DEFAULT_EMB_MODEL_PATH)
            state = torch.load(model_path, map_location=DEVICE)
            model, av_dim, phys_dim, lip_dim = _build_model_from_state(state, use_emb=use_emb)
            _models = [model]
            _AV_DIMS = [av_dim]
            _PHYS_DIMS = [phys_dim]
            _LIP_DIMS = [lip_dim]

            scaler_path = _resolve_single_scaler_path(model_path)
            if scaler_path.exists():
                _scalers = [joblib.load(scaler_path)]
            else:
                _scalers = [None]

        # Backward-compatible single-model aliases
        _model = _models[0]
        _AV_DIM = _AV_DIMS[0]
        _PHYS_DIM = _PHYS_DIMS[0]
        _LIP_DIM = _LIP_DIMS[0]
        _scaler = _scalers[0]
    except FileNotFoundError as exc:  # noqa: PERF203
        raise RuntimeError(
            "CFN model weights not found. Set CFN_ENSEMBLE_MANIFEST_PATH, "
            "CFN_ENSEMBLE_MODEL_PATHS, or CFN_EMB_MODEL_PATH."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load CFN model: {exc}") from exc

    if len(_scalers) != len(_models):
        raise RuntimeError("Scaler/model count mismatch in CFN artifacts.")
    if len(_AV_DIMS) != len(_models) or len(_PHYS_DIMS) != len(_models) or len(_LIP_DIMS) != len(_models):
        raise RuntimeError("Feature dimension metadata mismatch in CFN artifacts.")


def _prepare_features_for_model(base_av_vals, base_phys_vals, base_lip_vals, av_dim, phys_dim, lip_dim, scaler):
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
    lip_features = None
    if lip_dim and lip_dim > 0:
        lip_vals = list(base_lip_vals)
        if len(lip_vals) < lip_dim:
            lip_vals.extend([0.0] * (lip_dim - len(lip_vals)))
        if len(lip_vals) > lip_dim:
            lip_vals = lip_vals[:lip_dim]
        lip_features = np.array(lip_vals, dtype=np.float32)

    if scaler is not None:
        try:
            av_features = scaler["av"].transform([av_features])[0].astype(np.float32, copy=False)
            phys_features = scaler["phys"].transform([phys_features])[0].astype(np.float32, copy=False)
            if lip_features is not None and isinstance(scaler, dict) and "lip" in scaler:
                lip_features = scaler["lip"].transform([lip_features])[0].astype(np.float32, copy=False)
        except ValueError as exc:
            shape_key = (int(av_dim or -1), int(phys_dim or -1), int(lip_dim or -1), str(exc))
            if shape_key not in _scaler_shape_warned:
                warnings.warn(f"Skipping scaler due to shape mismatch: {exc}")
                _scaler_shape_warned.add(shape_key)
            av_features = av_features.astype(np.float32, copy=False)
            phys_features = phys_features.astype(np.float32, copy=False)
            if lip_features is not None:
                lip_features = lip_features.astype(np.float32, copy=False)

    # Some historical scalers contain near-zero variance on a few columns
    # (notably AV range-style features), which can explode z-scores and force
    # saturated 0/1 outputs across all frames. Bound the normalized features
    # before model inference to keep the active ensemble numerically useful.
    if _FEATURE_Z_CLIP > 0:
        av_features = np.clip(av_features, -_FEATURE_Z_CLIP, _FEATURE_Z_CLIP).astype(np.float32, copy=False)
        phys_features = np.clip(phys_features, -_FEATURE_Z_CLIP, _FEATURE_Z_CLIP).astype(np.float32, copy=False)
        if lip_features is not None:
            lip_features = np.clip(lip_features, -_FEATURE_Z_CLIP, _FEATURE_Z_CLIP).astype(np.float32, copy=False)

    return av_features, phys_features, lip_features


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


def _model_forward(model, av_tensor, phys_tensor, lip_tensor=None):
    """
    Run a CFN checkpoint while tolerating legacy call sites that still prepare
    an optional lip tensor. The current production models are 2-branch, but
    some historical checkpoints/callers still thread a third input through.
    """
    try:
        if lip_tensor is not None:
            return model(av_tensor, phys_tensor, lip_tensor)
        return model(av_tensor, phys_tensor)
    except TypeError:
        return model(av_tensor, phys_tensor)


def run_cfn_on_video(
    video_path,
    threshold=0.247707,
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
    runtime_temp = _resolve_runtime_temperature()

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

        chunk_results = []
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
                "effnet_b4_face_emb": 0.0,
                "lip_roi_emb": 0.0,
                "wav2vec2_base_ft_emb": 0.0,
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
            lip_map = {**av_map, **phys_map}
            base_lip_vals = _ordered_feature_values(lip_map, _LIP_FEATURE_ORDER)

            probs = []
            with torch.no_grad():
                for model, scaler, av_dim, phys_dim, lip_dim in zip(
                    _models,
                    _scalers,
                    _AV_DIMS,
                    _PHYS_DIMS,
                    _LIP_DIMS,
                ):
                    av_features, phys_features, lip_features = _prepare_features_for_model(
                        base_av_vals,
                        base_phys_vals,
                        base_lip_vals,
                        av_dim,
                        phys_dim,
                        lip_dim,
                        scaler,
                    )
                    X_av = torch.tensor(av_features).unsqueeze(0).to(DEVICE)
                    X_phys = torch.tensor(phys_features).unsqueeze(0).to(DEVICE)
                    X_lip = (
                        torch.tensor(lip_features).unsqueeze(0).to(DEVICE)
                        if lip_features is not None
                        else None
                    )
                    probs.append(_model_forward(model, X_av, X_phys, X_lip).item())
            prob = float(np.mean(probs)) if probs else 0.0

            bbox = None
            if include_bboxes and (prob >= threshold or (causal_threshold is not None and av_mismatch[i] >= causal_threshold)):
                bbox = mouth_bbox_from_landmarks(
                    frame.get("landmarks"),
                    frame["frame"].shape
                )
                if bbox is None:
                    bbox = detect_face_bbox(frame["frame"])

            chunk_results.append({
                "timestamp": frame["timestamp"],
                "fake_prob_raw": float(prob),
                "fake_prob": float(prob),
                "av_mismatch": float(av_mismatch[i]),
                "bbox": bbox
            })

        if chunk_results:
            chunk_probs = np.asarray([float(r.get("fake_prob_raw", 0.0)) for r in chunk_results], dtype=np.float32)
            if abs(float(runtime_temp) - 1.0) > 1e-6:
                chunk_probs = _apply_temperature_scale(chunk_probs, float(runtime_temp))

            t2a_temp = 1.0
            if bool(_T2A_ENABLE) and chunk_probs.size >= int(_T2A_MIN_FRAMES):
                t2a_temp = _estimate_t2a_temperature(
                    chunk_probs,
                    target_entropy=float(_T2A_TARGET_ENTROPY),
                    max_temp=float(_T2A_MAX_TEMP),
                )
                if abs(float(t2a_temp) - 1.0) > 1e-6:
                    chunk_probs = _apply_temperature_scale(chunk_probs, float(t2a_temp))

            for row, p_adj in zip(chunk_results, chunk_probs.tolist()):
                row["fake_prob"] = float(p_adj)
                row["temp_scale"] = float(runtime_temp)
                if bool(_T2A_ENABLE):
                    row["t2a_temp"] = float(t2a_temp)
                results.append(row)

        chunk_start += current_chunk

    return results
