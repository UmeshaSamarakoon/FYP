#!/usr/bin/env python3
"""
Train a lightweight video-level calibrator on cached frame outputs.

Input cache format (json list) is the same as mixed_domain_sweep.py:
  [{"path": ..., "label": 0/1, "dataset": "...", "probs": [...], "mism": [...]}]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cvi.video_calibrator_features import FEATURE_NAMES, build_video_feature_vector


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _normalize_causal_weights(av_weight: float, prob_weight: float, default_av: float = 0.65, default_prob: float = 0.35):
    """
    Normalize a pair of non-negative weights so they sum to 1.
    Falls back to default weights when the inputs are invalid.
    """
    try:
        av = float(av_weight)
        prob = float(prob_weight)
    except (TypeError, ValueError):
        return default_av, default_prob

    if av < 0 or prob < 0:
        return default_av, default_prob

    total = av + prob
    if total > 0:
        return av / total, prob / total
    return default_av, default_prob


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    rec = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    prec = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    bal = 0.5 * (rec + spec)
    return {
        "acc": acc,
        "bal_acc": bal,
        "f1": f1,
        "rec": rec,
        "spec": spec,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _extract_feature_rows(cache_rows):
    X, y, ds = [], [], []
    for row in cache_rows:
        probs = row.get("probs", []) or []
        mism = row.get("mism", []) or []
        n = min(len(probs), len(mism))
        if n <= 0:
            continue
        frames = [
            {"fake_prob_smooth": float(probs[i]), "av_mismatch": float(mism[i])}
            for i in range(n)
        ]
        x = build_video_feature_vector(frames, prob_key="fake_prob_smooth")
        X.append(x)
        y.append(int(row.get("label", 0)))
        ds.append(str(row.get("dataset", "Unknown")))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(ds, dtype=object)


def _per_dataset_metrics(y_true, y_pred, ds):
    out = {}
    for name in sorted(set(ds.tolist())):
        m = ds == name
        out[name] = _metrics(y_true[m], y_pred[m])
    return out


def _macro_bal(per_ds):
    if not per_ds:
        return 0.0
    return float(np.mean([v["bal_acc"] for v in per_ds.values()]))


def _build_model(model_c: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(model_c),
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _select_threshold(y_true, probs, ds, min_rec: float, min_spec: float):
    best = None
    for thr in np.linspace(0.05, 0.95, 181):
        y_hat = (np.asarray(probs, dtype=np.float64) >= float(thr)).astype(int)
        overall = _metrics(y_true, y_hat)
        if overall["rec"] < min_rec or overall["spec"] < min_spec:
            continue
        per_ds = _per_dataset_metrics(y_true, y_hat, ds)
        macro_bal = _macro_bal(per_ds)
        key = (macro_bal, overall["bal_acc"], overall["f1"])
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "threshold": float(thr),
                "overall": overall,
                "per_dataset": per_ds,
                "macro_bal_acc": macro_bal,
            }
    if best is None:
        raise RuntimeError("No threshold met the requested constraints.")
    return best


def main():
    parser = argparse.ArgumentParser(description="Train video-level CFN decision calibrator from cached frame outputs.")
    parser.add_argument("--cache", default="eval_cache_mixed.json", help="Input cache JSON from mixed_domain_sweep.py")
    parser.add_argument("--out", default="models/video_calibrator.pkl", help="Output calibrator file")
    parser.add_argument("--holdout-cache", default="", help="Optional held-out cache JSON for reporting only.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds used for threshold selection.")
    parser.add_argument("--model-c", type=float, default=2.0, help="Inverse regularization strength for LogisticRegression.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rec", type=float, default=0.0, help="Optional recall floor for threshold selection")
    parser.add_argument("--min-spec", type=float, default=0.0, help="Optional specificity floor for threshold selection")
    parser.add_argument(
        "--optimize",
        choices=["macro"],
        default="macro",
        help="Threshold objective: macro-balanced (only option).",
    )
    parser.add_argument(
        "--causal-av-weight",
        type=float,
        default=0.65,
        help="Base weight applied to av_mismatch when computing the causal breach score.",
    )
    parser.add_argument(
        "--causal-prob-weight",
        type=float,
        default=0.35,
        help="Base weight applied to fake_prob when computing the causal breach score.",
    )
    args = parser.parse_args()

    causal_av_weight, causal_prob_weight = _normalize_causal_weights(
        args.causal_av_weight, args.causal_prob_weight
    )

    cache_path = Path(args.cache)
    rows = json.loads(cache_path.read_text())
    X, y, ds = _extract_feature_rows(rows)
    if len(X) < 20:
        raise RuntimeError(f"Not enough usable rows in cache: {len(X)}")

    cv_folds = max(2, int(args.cv_folds))
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros(len(X), dtype=np.float64)
    for train_idx, val_idx in splitter.split(X, y):
        model = _build_model(model_c=args.model_c)
        model.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = model.predict_proba(X[val_idx])[:, 1]

    best = _select_threshold(
        y_true=y,
        probs=oof_probs,
        ds=ds,
        min_rec=float(args.min_rec),
        min_spec=float(args.min_spec),
    )

    model = _build_model(model_c=args.model_c)
    model.fit(X, y)

    holdout_metrics = None
    holdout_cache = str(args.holdout_cache).strip()
    if holdout_cache:
        holdout_rows = json.loads(Path(holdout_cache).read_text())
        X_holdout, y_holdout, ds_holdout = _extract_feature_rows(holdout_rows)
        if len(X_holdout):
            holdout_probs = model.predict_proba(X_holdout)[:, 1]
            holdout_pred = (holdout_probs >= float(best["threshold"])).astype(int)
            holdout_metrics = _metrics(y_holdout, holdout_pred)
            holdout_metrics["per_dataset"] = _per_dataset_metrics(y_holdout, holdout_pred, ds_holdout)

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": best["threshold"],
        "causal_breach_weights": {"av": causal_av_weight, "prob": causal_prob_weight},
        "validation": {
            "source": f"{cache_path.stem}_{cv_folds}fold_cv",
            "n_rows": int(len(X)),
            "macro_bal_acc": best["macro_bal_acc"],
            "overall": best["overall"],
            "per_dataset": best["per_dataset"],
        },
        "source_cache": str(cache_path.resolve()),
        "selected_model_name": f"logreg_c{float(args.model_c):g}",
    }
    if holdout_metrics is not None:
        payload["holdout_test"] = {
            k: v for k, v in holdout_metrics.items() if k != "per_dataset"
        }
        payload["holdout_test"]["per_dataset"] = holdout_metrics["per_dataset"]
        payload["source_holdout_cache"] = str(Path(holdout_cache).resolve())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)

    print("Saved calibrator:", out_path.resolve())
    print(
        "Causal breach blend:",
        f"av={causal_av_weight:.3f}",
        f"prob={causal_prob_weight:.3f}",
    )
    print(
        "Validation:",
        f"macro_bal_acc={best['macro_bal_acc']:.3f}",
        f"overall_bal_acc={best['overall']['bal_acc']:.3f}",
        f"overall_f1={best['overall']['f1']:.3f}",
        f"thr={best['threshold']:.3f}",
    )
    for ds_name, m in best["per_dataset"].items():
        print(
            f"[{ds_name}] bal_acc={m['bal_acc']:.3f} f1={m['f1']:.3f} rec={m['rec']:.3f} spec={m['spec']:.3f}"
        )
    print("Set env:")
    print(f"CFN_VIDEO_CALIBRATOR_PATH={out_path.resolve()}")
    print(f"CFN_CALIBRATOR_THRESH={best['threshold']:.4f}")
    if holdout_metrics is not None:
        print(
            "Holdout:",
            f"acc={holdout_metrics['acc']:.3f}",
            f"bal_acc={holdout_metrics['bal_acc']:.3f}",
            f"f1={holdout_metrics['f1']:.3f}",
        )


if __name__ == "__main__":
    main()
