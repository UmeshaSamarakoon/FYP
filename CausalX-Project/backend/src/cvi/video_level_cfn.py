from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import csv
import json
import os
from pathlib import Path
import re
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.cvi import frame_causal_extractor
from src.cvi.feature_schema import (
    BASELINE_AV_FEATURES,
    BASELINE_PHYS_FEATURES,
    EMBEDDING_AV_FEATURES,
    EXTENDED_AV_FEATURES,
    EXTENDED_PHYS_FEATURES,
)
from src.modules.causal_fusion import CausalFusionNetworkV2
from src.modules.temporal_conv import TemporalConvNet
from src.preprocessing.batch_feature_extractor import extract_causal_features


_MODULE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST_PATH = _MODULE_DIR / "models" / "fakeavceleb_best_step46_multiseed_manifest.json"
_DEFAULT_RUNTIME_CALIBRATION_PATH = _MODULE_DIR / "models" / "fakeavceleb_best_step46_runtime_calibration.json"
_DEFAULT_TABULAR_SCORER_PATH = _MODULE_DIR / "models" / "fakeavceleb_runtimeparity_plus_balanced_tabular.joblib"
_DEFAULT_STEP46_PRECOMPUTE_DIR = _MODULE_DIR / "outputs" / "step46_precompute"
_DEFAULT_TEMPORAL_PRECOMPUTE_DIR = _MODULE_DIR / "outputs" / "runtimeparity_temporal"
_DEVICE = torch.device("cpu")


def _dedupe_keep_order(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


STEP46_AV_COLUMNS = _dedupe_keep_order(
    list(BASELINE_AV_FEATURES) + list(EXTENDED_AV_FEATURES) + list(EMBEDDING_AV_FEATURES)
)
STEP46_PHYS_COLUMNS = _dedupe_keep_order(list(BASELINE_PHYS_FEATURES) + list(EXTENDED_PHYS_FEATURES))
RUNTIME_EXTRA_FEATURE_COLUMNS = [
    "audio_zcr_mean",
    "audio_zcr_std",
    "audio_centroid_mean",
    "audio_centroid_std",
    "audio_bandwidth_mean",
    "audio_bandwidth_std",
    "audio_rolloff_mean",
    "audio_rolloff_std",
    "audio_flatness_mean",
    "audio_flatness_std",
    "audio_mfcc1_mean",
    "audio_mfcc1_std",
    "audio_mfcc2_mean",
    "audio_mfcc2_std",
    "face_laplacian_mean",
    "face_laplacian_std",
    "face_edge_density_mean",
    "face_edge_density_std",
    "face_entropy_mean",
    "face_entropy_std",
    "face_blockiness_mean",
    "face_blockiness_std",
    "face_flicker_mean",
    "face_flicker_std",
    "mouth_laplacian_mean",
    "mouth_laplacian_std",
    "mouth_edge_density_mean",
    "mouth_edge_density_std",
    "mouth_entropy_mean",
    "mouth_entropy_std",
    "mouth_blockiness_mean",
    "mouth_blockiness_std",
    "mouth_flicker_mean",
    "mouth_flicker_std",
]
STEP46_FEATURE_COLUMNS = list(STEP46_AV_COLUMNS) + list(STEP46_PHYS_COLUMNS) + list(RUNTIME_EXTRA_FEATURE_COLUMNS)
STEP46_METADATA_COLUMNS = ["video_id", "label", "dataset", "video_fake", "audio_fake", "path"]
STEP46_SCHEMA_COLUMNS = list(STEP46_FEATURE_COLUMNS) + list(STEP46_METADATA_COLUMNS)

TEMPORAL_SEQUENCE_COLUMNS = [
    "frame_presence_ratio",
    "lip_aperture_mean",
    "lip_aperture_std",
    "audio_rms_mean",
    "audio_rms_std",
    "jitter_mean",
    "jitter_std",
    "lip_velocity_mean",
    "lip_velocity_std",
    "mouth_aspect_mean",
    "mouth_area_mean",
    "mouth_area_delta_mean",
    "mouth_asym_mean",
    "mouth_motion_mean",
    "mouth_motion_std",
    "av_mismatch_mean",
    "av_local_corr_mean",
    "av_local_corr_std_mean",
    "av_local_lag_mean",
    "av_peak_corr_mean",
    "av_peak_prominence_mean",
    "av_onset_corr_mean",
]


@dataclass(frozen=True)
class VideoLevelScore:
    video_fake: int
    fake_prob: float
    threshold: float
    decision_source: str
    model_mode: str
    artifact_csv_path: str | None = None
    vote_ratio: float | None = None
    fold_scores: dict[str, float] | None = None
    fold_thresholds: dict[str, float] | None = None


@dataclass(frozen=True)
class _ModelBundle:
    model_path: str
    scaler_path: str | None
    threshold: float | None
    temperature: float


@dataclass(frozen=True)
class _TabularBundle:
    scorer_path: str
    feature_columns: tuple[str, ...]
    threshold: float
    model_name: str


@dataclass(frozen=True)
class _TemporalBundle:
    scorer_path: str
    feature_columns: tuple[str, ...]
    threshold: float
    model_name: str


@dataclass(frozen=True)
class Step46PrecomputeArtifact:
    csv_path: str
    row: dict[str, object]


class RuntimeParityTemporalScorer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: tuple[int, ...] = (32, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.temporal = TemporalConvNet(input_dim, list(channels))
        hidden_dim = int(channels[-1]) if channels else int(input_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.temporal(x.transpose(1, 2))
        return self.head(feats).squeeze(1)


def _safe_float(value, default=0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _safe_slug(value: str, default: str = "video") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    slug = slug.strip("._-")
    return slug or default


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _window_mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [_safe_float(row.get(key), 0.0) for row in rows]
    return float(np.mean(vals)) if vals else 0.0


def _window_mean_std(rows: list[dict[str, object]], key: str) -> tuple[float, float]:
    vals = [_safe_float(row.get(key), 0.0) for row in rows]
    return _mean_std(vals)


def _ordered_feature_values(feature_map: dict[str, object], ordered_cols: list[str]) -> list[float]:
    values = []
    for col in ordered_cols:
        values.append(_safe_float(feature_map.get(col, 0.0), default=0.0))
    return values


def normalize_step46_row(
    feature_map: dict[str, object],
    *,
    video_path: str,
    label: int = -1,
    dataset: str = "live_upload",
    video_fake: int = -1,
    audio_fake: int = -1,
    video_id: str | None = None,
) -> dict[str, object]:
    path_obj = Path(video_path)
    row: dict[str, object] = {}
    for col in STEP46_FEATURE_COLUMNS:
        row[col] = _safe_float(feature_map.get(col, 0.0), default=0.0)
    row["video_id"] = str(video_id or path_obj.stem)
    row["label"] = int(label)
    row["dataset"] = str(dataset)
    row["video_fake"] = int(video_fake)
    row["audio_fake"] = int(audio_fake)
    row["path"] = str(path_obj)
    return row


def write_step46_artifact(
    row: dict[str, object],
    *,
    artifact_dir: str | Path | None = None,
    file_stem: str | None = None,
) -> Step46PrecomputeArtifact:
    out_dir = Path(artifact_dir) if artifact_dir is not None else _DEFAULT_STEP46_PRECOMPUTE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_slug(file_stem or str(row.get("video_id", "")) or Path(str(row.get("path", ""))).stem)
    ts = int(time.time() * 1000)
    csv_path = out_dir / f"{stem}_{ts}.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STEP46_SCHEMA_COLUMNS)
        writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in STEP46_SCHEMA_COLUMNS})

    return Step46PrecomputeArtifact(csv_path=str(csv_path.resolve()), row=dict(row))


def read_step46_artifact(csv_path: str | Path) -> dict[str, object] | None:
    path = Path(csv_path)
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        return None
    out: dict[str, object] = {}
    for col in STEP46_FEATURE_COLUMNS:
        out[col] = _safe_float(row.get(col, 0.0), default=0.0)
    for col in STEP46_METADATA_COLUMNS:
        out[col] = row.get(col)
    return out


def extract_runtimeparity_temporal_sequence(
    video_path: str | os.PathLike[str],
    *,
    duration_s: float = 6.0,
    target_fps: float = 12.0,
    window_s: float = 0.75,
    max_windows: int = 8,
) -> dict[str, object] | None:
    force_fallback = os.getenv("CFN_TEMPORAL_FORCE_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}
    if force_fallback:
        frame_causal_extractor.FACE_MESH = None

    frames = frame_causal_extractor.extract_frame_level_features(
        str(video_path),
        start_time=0.0,
        duration=float(duration_s),
        target_fps=float(target_fps),
        include_frame=False,
        include_landmarks=False,
    )
    if not frames or len(frames) < 6:
        return None

    sync = frame_causal_extractor.compute_av_sync_signals(frames, window=5)
    times = np.asarray([_safe_float(frame.get("timestamp"), 0.0) for frame in frames], dtype=np.float32)

    rows: list[dict[str, object]] = []
    for idx, frame in enumerate(frames):
        rows.append(
            {
                "timestamp": _safe_float(frame.get("timestamp"), 0.0),
                "lip_aperture": _safe_float(frame.get("lip_aperture"), 0.0),
                "audio_rms": _safe_float(frame.get("audio_rms"), 0.0),
                "jitter": _safe_float(frame.get("jitter"), 0.0),
                "jitter_std": _safe_float(frame.get("jitter_std"), 0.0),
                "lip_velocity": _safe_float(frame.get("lip_velocity"), 0.0),
                "mouth_aspect": _safe_float(frame.get("mouth_aspect"), 0.0),
                "mouth_area_norm": _safe_float(frame.get("mouth_area_norm"), 0.0),
                "mouth_area_delta": _safe_float(frame.get("mouth_area_delta"), 0.0),
                "mouth_asym": _safe_float(frame.get("mouth_asym"), 0.0),
                "mouth_motion": _safe_float(frame.get("mouth_motion"), 0.0),
                "mouth_motion_std": _safe_float(frame.get("mouth_motion_std"), 0.0),
                "av_mismatch": _safe_float(sync["mismatch"][idx], 0.0),
                "av_local_corr": _safe_float(sync["local_corr"][idx], 0.0),
                "av_local_corr_std": _safe_float(sync["local_corr_std"][idx], 0.0),
                "av_local_lag": _safe_float(sync["local_lag"][idx], 0.0),
                "av_peak_corr": _safe_float(sync["peak_corr"][idx], 0.0),
                "av_peak_prominence": _safe_float(sync["peak_prominence"][idx], 0.0),
                "av_onset_corr": _safe_float(sync["onset_corr"][idx], 0.0),
            }
        )

    seq = np.zeros((int(max_windows), len(TEMPORAL_SEQUENCE_COLUMNS)), dtype=np.float32)
    mask = np.zeros(int(max_windows), dtype=np.float32)
    starts = [float(i) * float(window_s) for i in range(int(max_windows))]

    for window_idx, start in enumerate(starts):
        end = start + float(window_s)
        bucket = [row for row in rows if start <= float(row["timestamp"]) < end]
        if not bucket:
            continue

        mask[window_idx] = 1.0
        expected = max(int(round(float(window_s) * float(target_fps))), 1)
        lip_mean, lip_std = _window_mean_std(bucket, "lip_aperture")
        audio_mean, audio_std = _window_mean_std(bucket, "audio_rms")
        jitter_mean, jitter_std = _window_mean_std(bucket, "jitter")
        lv_mean, lv_std = _window_mean_std(bucket, "lip_velocity")
        motion_mean, motion_std = _window_mean_std(bucket, "mouth_motion")
        seq[window_idx] = np.asarray(
            [
                float(min(len(bucket) / expected, 1.0)),
                lip_mean,
                lip_std,
                audio_mean,
                audio_std,
                jitter_mean,
                jitter_std,
                lv_mean,
                lv_std,
                _window_mean(bucket, "mouth_aspect"),
                _window_mean(bucket, "mouth_area_norm"),
                _window_mean(bucket, "mouth_area_delta"),
                _window_mean(bucket, "mouth_asym"),
                motion_mean,
                motion_std,
                _window_mean(bucket, "av_mismatch"),
                _window_mean(bucket, "av_local_corr"),
                _window_mean(bucket, "av_local_corr_std"),
                _window_mean(bucket, "av_local_lag"),
                _window_mean(bucket, "av_peak_corr"),
                _window_mean(bucket, "av_peak_prominence"),
                _window_mean(bucket, "av_onset_corr"),
            ],
            dtype=np.float32,
        )

    return {
        "video_path": str(Path(video_path)),
        "extractor_mode": ("fallback" if force_fallback or frame_causal_extractor.FACE_MESH is None else "facemesh"),
        "feature_columns": list(TEMPORAL_SEQUENCE_COLUMNS),
        "window_s": float(window_s),
        "max_windows": int(max_windows),
        "duration_s": float(duration_s),
        "target_fps": float(target_fps),
        "num_frames": int(len(rows)),
        "frame_timestamps": times.tolist(),
        "sequence": seq.tolist(),
        "mask": mask.tolist(),
    }


def _resolve_temporal_output_dir() -> Path:
    explicit = os.getenv("CFN_VIDEO_LEVEL_TEMPORAL_PRECOMPUTE_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return _DEFAULT_TEMPORAL_PRECOMPUTE_DIR


def write_runtimeparity_temporal_artifact(
    video_path: str | os.PathLike[str],
    *,
    out_dir: str | os.PathLike[str] | None = None,
    duration_s: float = 6.0,
    target_fps: float = 12.0,
    window_s: float = 0.75,
    max_windows: int = 8,
) -> str | None:
    payload = extract_runtimeparity_temporal_sequence(
        video_path,
        duration_s=float(duration_s),
        target_fps=float(target_fps),
        window_s=float(window_s),
        max_windows=int(max_windows),
    )
    if payload is None:
        return None

    base_dir = Path(out_dir).expanduser() if out_dir is not None else _resolve_temporal_output_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem or "clip"
    out_path = base_dir / f"{stem}_{int(time.time() * 1000)}.json"
    out_path.write_text(json.dumps(payload))
    return str(out_path.resolve())


def read_runtimeparity_temporal_artifact(path: str | os.PathLike[str]) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    sequence = np.asarray(payload.get("sequence", []), dtype=np.float32)
    mask = np.asarray(payload.get("mask", []), dtype=np.float32)
    payload["sequence"] = sequence
    payload["mask"] = mask
    return payload


def _load_threshold_from_model_dir(model_dir: Path) -> float | None:
    report_path = model_dir / "cfn_threshold_report.json"
    if not report_path.exists():
        return None
    try:
        obj = json.loads(report_path.read_text())
    except Exception:
        return None
    chosen = obj.get("chosen_epoch_report") or {}
    thr = chosen.get("selection_threshold")
    if thr is None:
        return None
    return _safe_float(thr, default=0.5)


def _load_temperature_from_model_dir(model_dir: Path) -> float:
    temp_path = model_dir / "cfn_temperature.json"
    if not temp_path.exists():
        return 1.0
    try:
        obj = json.loads(temp_path.read_text())
    except Exception:
        return 1.0
    return max(_safe_float(obj.get("temperature", 1.0), default=1.0), 1e-4)


def _resolve_tabular_scorer_path() -> Path | None:
    explicit = os.getenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    allow_default_raw = os.getenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", "").strip().lower()
    allow_default = allow_default_raw in {"1", "true", "yes", "on"}
    if allow_default and _DEFAULT_TABULAR_SCORER_PATH.exists():
        return _DEFAULT_TABULAR_SCORER_PATH
    return None


def _resolve_temporal_scorer_path() -> Path | None:
    explicit = os.getenv("CFN_VIDEO_LEVEL_TEMPORAL_SCORER_PATH", "").strip()
    if not explicit:
        return None
    p = Path(explicit).expanduser()
    return p if p.exists() else None


@lru_cache(maxsize=None)
def _load_model(model_path_str: str):
    model_path = Path(model_path_str)
    state = torch.load(model_path, map_location=_DEVICE)
    av_weight = state.get("av_branch.0.weight")
    phys_weight = state.get("physical_branch.0.weight")
    av_dim = int(av_weight.shape[1]) if isinstance(av_weight, torch.Tensor) and av_weight.ndim >= 2 else len(STEP46_AV_COLUMNS)
    phys_dim = int(phys_weight.shape[1]) if isinstance(phys_weight, torch.Tensor) and phys_weight.ndim >= 2 else len(STEP46_PHYS_COLUMNS)
    model = CausalFusionNetworkV2(
        av_dim=av_dim,
        phys_dim=phys_dim,
        enable_av_input_layernorm=any(str(k).startswith("av_input_ln.") for k in state.keys()),
    ).to(_DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, av_dim, phys_dim


@lru_cache(maxsize=None)
def _load_scaler(scaler_path_str: str | None):
    if not scaler_path_str:
        return None
    scaler_path = Path(scaler_path_str)
    if not scaler_path.exists():
        return None
    return joblib.load(scaler_path)


@lru_cache(maxsize=1)
def _load_tabular_scorer():
    scorer_path = _resolve_tabular_scorer_path()
    if scorer_path is None:
        return None
    try:
        payload = joblib.load(scorer_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    feature_columns = payload.get("feature_columns")
    if model is None or not isinstance(feature_columns, (list, tuple)) or not feature_columns:
        return None
    threshold = _safe_float(payload.get("threshold", 0.5), default=0.5)
    model_name = str(payload.get("model_name", "tabular")).strip() or "tabular"
    return {
        "bundle": _TabularBundle(
            scorer_path=str(scorer_path.resolve()),
            feature_columns=tuple(str(c) for c in feature_columns),
            threshold=float(threshold),
            model_name=model_name,
        ),
        "model": model,
    }


@lru_cache(maxsize=1)
def _load_temporal_scorer():
    scorer_path = _resolve_temporal_scorer_path()
    if scorer_path is None:
        return None
    try:
        payload = torch.load(scorer_path, map_location=_DEVICE, weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    feature_columns = payload.get("feature_columns")
    model_config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(feature_columns, list) or not isinstance(model_config, dict) or not isinstance(state, dict):
        return None
    try:
        model = RuntimeParityTemporalScorer(
            input_dim=int(model_config.get("input_dim", len(feature_columns))),
            channels=tuple(int(v) for v in model_config.get("channels", [32, 32])),
            dropout=float(model_config.get("dropout", 0.2)),
        ).to(_DEVICE)
        model.load_state_dict(state)
        model.eval()
    except Exception:
        return None
    threshold = _safe_float(payload.get("threshold", 0.5), default=0.5)
    return {
        "bundle": _TemporalBundle(
            scorer_path=str(scorer_path.resolve()),
            feature_columns=tuple(str(c) for c in feature_columns),
            threshold=float(threshold),
            model_name="runtimeparity_temporal",
        ),
        "model": model,
        "mean": np.asarray(payload.get("normalization_mean", []), dtype=np.float32),
        "std": np.asarray(payload.get("normalization_std", []), dtype=np.float32),
    }


def _score_bundle(bundle: _ModelBundle, feature_map: dict[str, object]) -> float | None:
    from src.cvi.cfn_frame_inference import _apply_temperature_scale, _prepare_features_for_model

    try:
        model, av_dim, phys_dim = _load_model(bundle.model_path)
        scaler = _load_scaler(bundle.scaler_path)
    except Exception:
        return None
    base_av = _ordered_feature_values(feature_map, STEP46_AV_COLUMNS)
    base_phys = _ordered_feature_values(feature_map, STEP46_PHYS_COLUMNS)
    av_features, phys_features, _ = _prepare_features_for_model(
        base_av,
        base_phys,
        [],
        av_dim,
        phys_dim,
        0,
        scaler,
    )
    av_tensor = torch.tensor(av_features, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    phys_tensor = torch.tensor(phys_features, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        prob = float(model(av_tensor, phys_tensor).item())
    if not np.isfinite(prob):
        return None
    if abs(float(bundle.temperature) - 1.0) > 1e-6:
        prob = float(_apply_temperature_scale(np.array([prob], dtype=np.float32), float(bundle.temperature))[0])
    return _safe_float(prob, default=0.0)


def _score_tabular_bundle(bundle: _TabularBundle, feature_map: dict[str, object]) -> float | None:
    payload = _load_tabular_scorer()
    if payload is None:
        return None
    model = payload.get("model")
    if model is None:
        return None
    values = pd.DataFrame(
        [
            {
                col: _safe_float(feature_map.get(col, 0.0), default=0.0)
                for col in bundle.feature_columns
            }
        ],
        columns=list(bundle.feature_columns),
        dtype=np.float32,
    )
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(values)[0, 1])
        elif hasattr(model, "decision_function"):
            margin = float(model.decision_function(values)[0])
            prob = float(1.0 / (1.0 + np.exp(-margin)))
        else:
            return None
    except Exception:
        return None
    return _safe_float(prob, default=0.0)


def _resolve_manifest_path() -> Path | None:
    explicit = os.getenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    allow_default_raw = os.getenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", "").strip().lower()
    allow_default = allow_default_raw in {"1", "true", "yes", "on"}
    if allow_default and _DEFAULT_MANIFEST_PATH.exists():
        return _DEFAULT_MANIFEST_PATH
    return None


def _resolve_single_model_dir() -> Path | None:
    explicit = os.getenv("CFN_VIDEO_LEVEL_MODEL_DIR", "").strip()
    if not explicit:
        return None
    model_dir = Path(explicit).expanduser()
    if (model_dir / "cfn_emb.pth").exists():
        return model_dir
    return None


def _bundle_from_model_dir(model_dir: Path, threshold_override: float | None = None) -> _ModelBundle | None:
    model_dir = model_dir.expanduser()
    model_path = model_dir / "cfn_emb.pth"
    if not model_path.exists():
        return None
    threshold = threshold_override
    if threshold is None:
        threshold = _load_threshold_from_model_dir(model_dir)
    scaler_path = model_dir / "cfn_scaler.pkl"
    return _ModelBundle(
        model_path=str(model_path.resolve()),
        scaler_path=str(scaler_path.resolve()) if scaler_path.exists() else None,
        threshold=threshold,
        temperature=_load_temperature_from_model_dir(model_dir),
    )


@lru_cache(maxsize=1)
def _load_runtime_calibration():
    explicit = os.getenv("CFN_VIDEO_LEVEL_RUNTIME_CALIBRATION_JSON", "").strip()
    if explicit:
        calibration_path = Path(explicit).expanduser()
    else:
        calibration_path = _DEFAULT_RUNTIME_CALIBRATION_PATH
    if not calibration_path.exists():
        return None

    try:
        payload = json.loads(calibration_path.read_text())
    except Exception:
        return None

    threshold = payload.get("ensemble_threshold")
    if threshold is None:
        return None
    mode = str(payload.get("threshold_mode", "mean_prob")).strip().lower()
    if mode != "mean_prob":
        return None
    return {
        "path": str(calibration_path),
        "threshold_mode": mode,
        "ensemble_threshold": _safe_float(threshold, default=0.5),
        "source_manifest": str(payload.get("source_manifest", "")).strip(),
    }


@lru_cache(maxsize=1)
def _load_selection_bundle():
    explicit = os.getenv("CFN_VIDEO_LEVEL_SELECTION_JSON", "").strip()
    if not explicit:
        return None

    selection_path = Path(explicit).expanduser()
    if not selection_path.exists():
        return None

    try:
        payload = json.loads(selection_path.read_text())
    except Exception:
        return None

    best = payload.get("best_model")
    if not isinstance(best, dict):
        return None

    raw_model_dir = str(best.get("model_dir", "")).strip()
    if not raw_model_dir:
        return None

    threshold = None
    for key in (
        "clean_threshold",
        "clean_best_f1_threshold",
        "clean_report_threshold",
        "selection_threshold",
    ):
        if key in best:
            threshold = _safe_float(best.get(key), default=0.5)
            break

    return _bundle_from_model_dir(Path(raw_model_dir), threshold_override=threshold)


@lru_cache(maxsize=1)
def _load_manifest_spec():
    manifest_path = _resolve_manifest_path()
    if manifest_path is None:
        return None

    payload = json.loads(manifest_path.read_text())
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return None

    bundles_by_fold: dict[str, list[_ModelBundle]] = {}
    thresholds_by_fold: dict[str, float] = {}

    summary_rel = str(payload.get("summary_path", "")).strip()
    if summary_rel:
        summary_path = (manifest_path.parent / summary_rel).resolve()
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                csv_path = Path(str(summary.get("artifacts", {}).get("ensemble_fold_metrics_csv", "")).strip())
                if csv_path.exists():
                    with csv_path.open("r", newline="") as f:
                        for row in csv.DictReader(f):
                            fold = str(row.get("fold", "")).strip()
                            if not fold:
                                continue
                            thresholds_by_fold[fold] = _safe_float(row.get("eval_threshold"), default=0.5)
            except Exception:
                thresholds_by_fold = {}

    for entry in artifacts:
        if not isinstance(entry, dict):
            continue
        model_path = entry.get("model_path")
        if not model_path:
            continue
        scaler_path = entry.get("scaler_path")
        resolved_model = Path(model_path)
        if not resolved_model.is_absolute():
            resolved_model = (manifest_path.parent / resolved_model).resolve()
        if not resolved_model.exists():
            continue
        resolved_scaler = None
        if scaler_path:
            resolved_scaler = Path(scaler_path)
            if not resolved_scaler.is_absolute():
                resolved_scaler = (manifest_path.parent / resolved_scaler).resolve()
            if not resolved_scaler.exists():
                resolved_scaler = None
        fold = str(entry.get("fold", "all")).strip() or "all"
        model_dir = resolved_model.parent
        bundle = _ModelBundle(
            model_path=str(resolved_model),
            scaler_path=str(resolved_scaler) if resolved_scaler is not None else None,
            threshold=_load_threshold_from_model_dir(model_dir),
            temperature=_load_temperature_from_model_dir(model_dir),
        )
        bundles_by_fold.setdefault(fold, []).append(bundle)

    if not bundles_by_fold:
        return None

    return {
        "manifest_name": str(payload.get("name", "video_level_manifest")),
        "bundles_by_fold": bundles_by_fold,
        "thresholds_by_fold": thresholds_by_fold,
    }


def _mean_threshold(bundles: list[_ModelBundle], default: float = 0.5) -> float:
    vals = [float(b.threshold) for b in bundles if b.threshold is not None and np.isfinite(float(b.threshold))]
    if not vals:
        return float(default)
    return float(np.mean(vals))


def _resolve_precompute_dir() -> Path | None:
    explicit = os.getenv("CFN_VIDEO_LEVEL_PRECOMPUTE_DIR", "").strip()
    if not explicit:
        return None
    return Path(explicit).expanduser()


def _has_video_level_scorer() -> bool:
    if _load_temporal_scorer() is not None:
        return True
    if _load_tabular_scorer() is not None:
        return True
    if _load_selection_bundle() is not None:
        return True
    if _resolve_single_model_dir() is not None:
        return True
    return _load_manifest_spec() is not None


def _score_video_level_feature_map(
    feature_map: dict[str, object],
    *,
    artifact_csv_path: str | None = None,
) -> VideoLevelScore | None:
    tabular_payload = _load_tabular_scorer()
    selection_bundle = _load_selection_bundle()
    single_model_dir = _resolve_single_model_dir()
    runtime_calibration = _load_runtime_calibration()
    spec = None
    if tabular_payload is None and selection_bundle is None and single_model_dir is None:
        spec = _load_manifest_spec()
        if spec is None:
            return None

    if not feature_map:
        return None

    if tabular_payload is not None:
        bundle = tabular_payload["bundle"]
        prob = _score_tabular_bundle(bundle, feature_map)
        if prob is None:
            return None
        threshold = float(bundle.threshold)
        return VideoLevelScore(
            video_fake=int(prob >= threshold),
            fake_prob=float(prob),
            threshold=threshold,
            decision_source="video_level_tabular",
            model_mode="tabular",
            artifact_csv_path=artifact_csv_path,
        )

    if selection_bundle is not None:
        prob = _score_bundle(selection_bundle, feature_map)
        if prob is None:
            return None
        threshold = float(selection_bundle.threshold) if selection_bundle.threshold is not None else 0.5
        return VideoLevelScore(
            video_fake=int(prob >= threshold),
            fake_prob=float(prob),
            threshold=float(threshold),
            decision_source="video_level_cfn_selection",
            model_mode="selection",
            artifact_csv_path=artifact_csv_path,
        )

    if single_model_dir is not None:
        bundle = _bundle_from_model_dir(single_model_dir)
        if bundle is None:
            return None
        prob = _score_bundle(bundle, feature_map)
        if prob is None:
            return None
        threshold = float(bundle.threshold) if bundle.threshold is not None else 0.5
        return VideoLevelScore(
            video_fake=int(prob >= threshold),
            fake_prob=float(prob),
            threshold=float(threshold),
            decision_source="video_level_cfn_single",
            model_mode="single",
            artifact_csv_path=artifact_csv_path,
        )

    fold_scores: dict[str, float] = {}
    fold_thresholds: dict[str, float] = {}
    fold_preds: list[int] = []
    all_bundle_scores: list[float] = []

    for fold_name, bundles in spec["bundles_by_fold"].items():
        scores = [s for s in (_score_bundle(bundle, feature_map) for bundle in bundles) if s is not None and np.isfinite(float(s))]
        if not scores:
            continue
        fold_score = float(np.mean(scores))
        fold_threshold = spec["thresholds_by_fold"].get(fold_name)
        if fold_threshold is None:
            fold_threshold = _mean_threshold(bundles, default=0.5)
        fold_scores[str(fold_name)] = fold_score
        fold_thresholds[str(fold_name)] = float(fold_threshold)
        fold_preds.append(int(fold_score >= float(fold_threshold)))
        all_bundle_scores.extend(scores)

    if not fold_scores and not all_bundle_scores:
        return None

    fake_prob = float(np.mean(list(fold_scores.values()) if fold_scores else all_bundle_scores))
    if fold_preds:
        vote_ratio = float(np.mean(fold_preds))
        threshold = float(np.mean(list(fold_thresholds.values()))) if fold_thresholds else 0.5
        decision_source = "video_level_cfn_ensemble"
        if runtime_calibration is not None:
            threshold = float(runtime_calibration["ensemble_threshold"])
            video_fake = int(fake_prob >= threshold)
            decision_source = "video_level_cfn_ensemble_calibrated"
        else:
            video_fake = int(vote_ratio >= 0.5)
            decision_source = "video_level_cfn_ensemble"
    else:
        vote_ratio = None
        threshold = 0.5
        video_fake = int(fake_prob >= threshold)
        decision_source = "video_level_cfn_ensemble"

    return VideoLevelScore(
        video_fake=int(video_fake),
        fake_prob=float(fake_prob),
        threshold=float(threshold),
        decision_source=decision_source,
        model_mode="ensemble",
        artifact_csv_path=artifact_csv_path,
        vote_ratio=vote_ratio,
        fold_scores=fold_scores or None,
        fold_thresholds=fold_thresholds or None,
    )


def _score_temporal_artifact(artifact_path: str | os.PathLike[str]) -> VideoLevelScore | None:
    payload = _load_temporal_scorer()
    if payload is None:
        return None
    artifact = read_runtimeparity_temporal_artifact(artifact_path)
    sequence = np.asarray(artifact.get("sequence", []), dtype=np.float32)
    if sequence.ndim != 2 or sequence.size == 0:
        return None
    mean = np.asarray(payload.get("mean", []), dtype=np.float32)
    std = np.asarray(payload.get("std", []), dtype=np.float32)
    if mean.size == sequence.shape[1] and std.size == sequence.shape[1]:
        std = np.where(std < 1e-6, 1.0, std)
        sequence = (sequence - mean[None, :]) / std[None, :]
    model = payload.get("model")
    if model is None:
        return None
    with torch.no_grad():
        logits = model(torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(_DEVICE))
        prob = float(torch.sigmoid(logits).item())
    bundle = payload["bundle"]
    threshold = float(bundle.threshold)
    return VideoLevelScore(
        video_fake=int(prob >= threshold),
        fake_prob=prob,
        threshold=threshold,
        decision_source="video_level_temporal",
        model_mode="temporal",
        artifact_csv_path=str(Path(artifact_path).resolve()),
    )


def create_step46_precompute_artifact(
    video_path: str | os.PathLike[str],
    *,
    label: int = -1,
    dataset: str = "live_upload",
    video_fake: int = -1,
    audio_fake: int = -1,
    video_id: str | None = None,
) -> str | None:
    feature_map = extract_causal_features(str(video_path))
    if not feature_map:
        return None
    row = normalize_step46_row(
        feature_map,
        video_path=str(video_path),
        label=label,
        dataset=dataset,
        video_fake=video_fake,
        audio_fake=audio_fake,
        video_id=video_id,
    )
    artifact = write_step46_artifact(
        row,
        artifact_dir=_resolve_precompute_dir(),
        file_stem=row.get("video_id"),
    )
    return artifact.csv_path


def score_video_level_precomputed_csv(csv_path: str | os.PathLike[str]) -> VideoLevelScore | None:
    row = read_step46_artifact(csv_path)
    if row is None:
        return None
    return _score_video_level_feature_map(row, artifact_csv_path=str(Path(csv_path).resolve()))


def score_video_level_cfn(video_path: str | os.PathLike[str]) -> VideoLevelScore | None:
    if not _has_video_level_scorer():
        return None
    if _load_temporal_scorer() is not None:
        artifact_path = write_runtimeparity_temporal_artifact(video_path)
        if artifact_path is None:
            return None
        return _score_temporal_artifact(artifact_path)
    artifact_csv_path = create_step46_precompute_artifact(video_path)
    if artifact_csv_path is None:
        return None
    return score_video_level_precomputed_csv(artifact_csv_path)
