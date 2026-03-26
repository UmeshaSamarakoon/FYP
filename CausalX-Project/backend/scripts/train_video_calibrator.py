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

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "prob_mean",
    "prob_std",
    "prob_p90",
    "prob_p95",
    "prob_max",
    "mism_mean",
    "mism_std",
    "mism_p90",
    "mism_p95",
    "mism_max",
    "prob_mism_corr",
    "ratio_prob_ge_0_70",
    "ratio_prob_ge_0_80",
    "ratio_mism_ge_0_70",
    "ratio_mism_ge_0_80",
]


def build_video_feature_vector(frames, prob_key="fake_prob"):
    if not frames:
        return np.zeros(15, dtype=np.float32)

    probs = np.array([f.get(prob_key, 0.0) for f in frames], dtype=np.float32)
    mism = np.array([f.get("av_mismatch", 0.0) for f in frames], dtype=np.float32)
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    mism = np.nan_to_num(mism, nan=0.0, posinf=1.0, neginf=0.0)

    def _stats(x):
        return (
            float(np.mean(x)),
            float(np.std(x)),
            float(np.percentile(x, 90)),
            float(np.percentile(x, 95)),
            float(np.max(x)),
        )

    p_mean, p_std, p_p90, p_p95, p_max = _stats(probs)
    m_mean, m_std, m_p90, m_p95, m_max = _stats(mism)

    p_std_val = float(np.std(probs))
    m_std_val = float(np.std(mism))
    if len(probs) > 1 and p_std_val > 1e-8 and m_std_val > 1e-8:
        p_center = probs - float(np.mean(probs))
        m_center = mism - float(np.mean(mism))
        corr = float(np.mean(p_center * m_center) / (p_std_val * m_std_val))
        if not np.isfinite(corr):
            corr = 0.0
    else:
        corr = 0.0

    ratio_p70 = float(np.mean(probs >= 0.70))
    ratio_p80 = float(np.mean(probs >= 0.80))
    ratio_m70 = float(np.mean(mism >= 0.70))
    ratio_m80 = float(np.mean(mism >= 0.80))

    return np.array(
        [
            p_mean, p_std, p_p90, p_p95, p_max,
            m_mean, m_std, m_p90, m_p95, m_max,
            corr,
            ratio_p70, ratio_p80,
            ratio_m70, ratio_m80,
        ],
        dtype=np.float32,
    )


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


def main():
    parser = argparse.ArgumentParser(description="Train video-level CFN decision calibrator from cached frame outputs.")
    parser.add_argument("--cache", default="eval_cache_mixed.json", help="Input cache JSON from mixed_domain_sweep.py")
    parser.add_argument("--out", default="models/video_calibrator.pkl", help="Output calibrator file")
    parser.add_argument("--test-size", type=float, default=0.25, help="Validation split fraction")
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

    strat = np.array([f"{d}:{yy}" for d, yy in zip(ds, y)], dtype=object)
    unique, counts = np.unique(strat, return_counts=True)
    if np.min(counts) < 2:
        strat = y

    X_train, X_val, y_train, y_val, ds_train, ds_val = train_test_split(
        X, y, ds,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=strat,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]

    best = None
    for thr in np.linspace(0.05, 0.95, 181):
        y_hat = (p_val >= thr).astype(int)
        overall = _metrics(y_val, y_hat)
        if overall["rec"] < args.min_rec or overall["spec"] < args.min_spec:
            continue
        per_ds = _per_dataset_metrics(y_val, y_hat, ds_val)
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

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": best["threshold"],
        "causal_breach_weights": {"av": causal_av_weight, "prob": causal_prob_weight},
        "validation": {
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "macro_bal_acc": best["macro_bal_acc"],
            "overall": best["overall"],
            "per_dataset": best["per_dataset"],
        },
        "source_cache": str(cache_path.resolve()),
    }
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


if __name__ == "__main__":
    main()
