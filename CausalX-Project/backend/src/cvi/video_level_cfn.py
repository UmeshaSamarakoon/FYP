from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.cvi.runtimeparity_temporal import read_runtimeparity_temporal_artifact, write_runtimeparity_temporal_artifact
from src.cvi.step46_precompute import (
    STEP46_AV_COLUMNS,
    STEP46_PHYS_COLUMNS,
    normalize_step46_row,
    read_step46_artifact,
    write_step46_artifact,
)
from src.modules.causal_fusion import CausalFusionNetworkV2
from src.modules.runtimeparity_temporal_model import RuntimeParityTemporalScorer
from src.preprocessing.batch_feature_extractor import extract_causal_features


_MODULE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST_PATH = _MODULE_DIR / "models" / "fakeavceleb_best_step46_multiseed_manifest.json"
_DEFAULT_RUNTIME_CALIBRATION_PATH = _MODULE_DIR / "models" / "fakeavceleb_best_step46_runtime_calibration.json"
_DEFAULT_TABULAR_SCORER_PATH = _MODULE_DIR / "models" / "fakeavceleb_runtimeparity_plus_balanced_tabular.joblib"
_DEVICE = torch.device("cpu")


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


def _safe_float(value, default=0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _ordered_feature_values(feature_map: dict[str, object], ordered_cols: list[str]) -> list[float]:
    values = []
    for col in ordered_cols:
        values.append(_safe_float(feature_map.get(col, 0.0), default=0.0))
    return values


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
    if allow_default_raw in {"0", "false", "no", "off"}:
        return None
    allow_default = allow_default_raw in {"", "1", "true", "yes", "on"}
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

    model, av_dim, phys_dim = _load_model(bundle.model_path)
    scaler = _load_scaler(bundle.scaler_path)
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
    if allow_default_raw in {"0", "false", "no", "off"}:
        return None
    allow_default = allow_default_raw in {"", "1", "true", "yes", "on"}
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
        resolved_scaler = None
        if scaler_path:
            resolved_scaler = Path(scaler_path)
            if not resolved_scaler.is_absolute():
                resolved_scaler = (manifest_path.parent / resolved_scaler).resolve()
        fold = str(entry.get("fold", "all")).strip() or "all"
        model_dir = resolved_model.parent
        bundle = _ModelBundle(
            model_path=str(resolved_model),
            scaler_path=str(resolved_scaler) if resolved_scaler is not None else None,
            threshold=_load_threshold_from_model_dir(model_dir),
            temperature=_load_temperature_from_model_dir(model_dir),
        )
        bundles_by_fold.setdefault(fold, []).append(bundle)

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
