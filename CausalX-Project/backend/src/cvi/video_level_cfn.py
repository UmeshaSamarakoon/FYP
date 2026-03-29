from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch

from src.cvi.cfn_frame_inference import (
    _AV_FEATURE_ORDER,
    _PHYS_FEATURE_ORDER,
    _apply_temperature_scale,
    _prepare_features_for_model,
)
from src.modules.causal_fusion import CausalFusionNetworkV2
from src.preprocessing.batch_feature_extractor import extract_causal_features


_MODULE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST_PATH = _MODULE_DIR / "models" / "fakeavceleb_best_step46_multiseed_manifest.json"
_DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class VideoLevelScore:
    video_fake: int
    fake_prob: float
    threshold: float
    decision_source: str
    model_mode: str
    vote_ratio: float | None = None
    fold_scores: dict[str, float] | None = None
    fold_thresholds: dict[str, float] | None = None


@dataclass(frozen=True)
class _ModelBundle:
    model_path: str
    scaler_path: str | None
    threshold: float | None
    temperature: float


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


@lru_cache(maxsize=None)
def _load_model(model_path_str: str):
    model_path = Path(model_path_str)
    state = torch.load(model_path, map_location=_DEVICE)
    av_weight = state.get("av_branch.0.weight")
    phys_weight = state.get("physical_branch.0.weight")
    av_dim = int(av_weight.shape[1]) if isinstance(av_weight, torch.Tensor) and av_weight.ndim >= 2 else len(_AV_FEATURE_ORDER)
    phys_dim = int(phys_weight.shape[1]) if isinstance(phys_weight, torch.Tensor) and phys_weight.ndim >= 2 else len(_PHYS_FEATURE_ORDER)
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


def _score_bundle(bundle: _ModelBundle, feature_map: dict[str, object]) -> float | None:
    model, av_dim, phys_dim = _load_model(bundle.model_path)
    scaler = _load_scaler(bundle.scaler_path)
    base_av = _ordered_feature_values(feature_map, _AV_FEATURE_ORDER)
    base_phys = _ordered_feature_values(feature_map, _PHYS_FEATURE_ORDER)
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


def _resolve_manifest_path() -> Path | None:
    explicit = os.getenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    allow_default = os.getenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", "false").strip().lower() == "true"
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


def score_video_level_cfn(video_path: str | os.PathLike[str]) -> VideoLevelScore | None:
    selection_bundle = _load_selection_bundle()
    single_model_dir = _resolve_single_model_dir()
    spec = None
    if selection_bundle is None and single_model_dir is None:
        spec = _load_manifest_spec()
        if spec is None:
            return None

    feature_map = extract_causal_features(str(video_path))
    if not feature_map:
        return None

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
        video_fake = int(vote_ratio >= 0.5)
        threshold = float(np.mean(list(fold_thresholds.values()))) if fold_thresholds else 0.5
    else:
        vote_ratio = None
        threshold = 0.5
        video_fake = int(fake_prob >= threshold)

    return VideoLevelScore(
        video_fake=int(video_fake),
        fake_prob=float(fake_prob),
        threshold=float(threshold),
        decision_source="video_level_cfn_ensemble",
        model_mode="ensemble",
        vote_ratio=vote_ratio,
        fold_scores=fold_scores or None,
        fold_thresholds=fold_thresholds or None,
    )
