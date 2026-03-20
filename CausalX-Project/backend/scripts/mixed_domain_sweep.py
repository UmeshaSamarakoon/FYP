#!/usr/bin/env python3
"""
Build a mixed-domain eval manifest (FakeAVCeleb + DFDC) and run a cached
threshold sweep optimized for macro balanced accuracy across datasets.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalRow:
    path: str
    label: int
    dataset: str


def ensure_src_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def build_balanced_subset(items, per_class_limit, seed):
    rng = random.Random(seed)
    fake = [x for x in items if int(x["label"]) == 1]
    real = [x for x in items if int(x["label"]) == 0]
    n = min(len(fake), len(real), per_class_limit)
    rng.shuffle(fake)
    rng.shuffle(real)
    out = fake[:n] + real[:n]
    rng.shuffle(out)
    return out


def build_manifest(root: Path, fakeav_root: Path, dfdc_root: Path, per_class_limit: int, seed: int, out_path: Path):
    from src.utils.dataset_registry import get_fakeavceleb_videos, get_dfdc_videos

    fakeav = get_fakeavceleb_videos(str(fakeav_root))
    dfdc = get_dfdc_videos(str(dfdc_root))

    fakeav_bal = build_balanced_subset(fakeav, per_class_limit=per_class_limit, seed=seed)
    dfdc_bal = build_balanced_subset(dfdc, per_class_limit=per_class_limit, seed=seed)

    rows = []
    for x in fakeav_bal:
        rows.append(EvalRow(path=str(Path(x["path"]).resolve()), label=int(x["label"]), dataset="FakeAVCeleb"))
    for x in dfdc_bal:
        rows.append(EvalRow(path=str(Path(x["path"]).resolve()), label=int(x["label"]), dataset="DFDC"))

    rng = random.Random(seed)
    rng.shuffle(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(f"{r.path}\t{r.label}\t{r.dataset}\n")
    return rows


def parse_manifest(path: Path):
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            p, y, ds = parts
        elif len(parts) == 2:
            p, y = parts
            lpath = p.lower()
            ds = "DFDC" if "dfdc" in lpath else ("FakeAVCeleb" if "fakeavceleb" in lpath else "Unknown")
        else:
            continue
        rows.append(EvalRow(path=p, label=int(y), dataset=ds))
    return rows


def conf_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    bal = 0.5 * (rec + spec)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "spec": spec,
        "bal_acc": bal,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def eval_combo(cache, prob_t, ratio_t, causal_t, require_flag):
    by_ds = {}
    total_tp = total_tn = total_fp = total_fn = 0

    for item in cache:
        ds = item["dataset"]
        if ds not in by_ds:
            by_ds[ds] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        y = int(item["label"])
        probs = np.array(item["probs"], dtype=float)
        mism = np.array(item["mism"], dtype=float)

        if probs.size == 0:
            pred = 0
        else:
            if require_flag:
                suspicious = (probs >= prob_t) & (mism >= causal_t)
            else:
                suspicious = (probs >= prob_t) | (mism >= causal_t)
            pred = int(float(np.mean(suspicious)) >= ratio_t)

        if pred == 1 and y == 1:
            by_ds[ds]["tp"] += 1
            total_tp += 1
        elif pred == 0 and y == 0:
            by_ds[ds]["tn"] += 1
            total_tn += 1
        elif pred == 1 and y == 0:
            by_ds[ds]["fp"] += 1
            total_fp += 1
        else:
            by_ds[ds]["fn"] += 1
            total_fn += 1

    overall = conf_metrics(total_tp, total_tn, total_fp, total_fn)
    per_dataset = {ds: conf_metrics(v["tp"], v["tn"], v["fp"], v["fn"]) for ds, v in by_ds.items()}
    macro_bal = float(np.mean([m["bal_acc"] for m in per_dataset.values()])) if per_dataset else 0.0
    macro_f1 = float(np.mean([m["f1"] for m in per_dataset.values()])) if per_dataset else 0.0

    return {"overall": overall, "per_dataset": per_dataset, "macro_bal_acc": macro_bal, "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser(description="Mixed-domain cached threshold sweep.")
    parser.add_argument("--manifest", default="eval_manifest_mixed.tsv")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--fakeav-root", default="data/raw/fakeavceleb")
    parser.add_argument("--dfdc-root", default="data/raw/dfdc")
    parser.add_argument("--per-class-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default="eval_cache_mixed.json")
    parser.add_argument("--out-json", default="models/mixed_sweep_results.json")
    parser.add_argument("--prob-grid", default="0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86")
    parser.add_argument("--ratio-grid", default="0.45,0.48,0.50,0.52,0.55,0.58,0.60,0.62,0.65")
    parser.add_argument("--causal-grid", default="0.65,0.68,0.70,0.72,0.75,0.78,0.80")
    parser.add_argument("--require-flag-options", default="false,true")
    parser.add_argument("--min-rec", type=float, default=0.0, help="Minimum overall recall constraint.")
    parser.add_argument("--min-spec", type=float, default=0.0, help="Minimum overall specificity constraint.")
    parser.add_argument("--min-domain-rec", type=float, default=0.0, help="Minimum recall required for every dataset.")
    parser.add_argument("--min-domain-spec", type=float, default=0.0, help="Minimum specificity required for every dataset.")
    parser.add_argument(
        "--selection-objective",
        type=str,
        default="hybrid_robust",
        choices=["macro", "worst", "hybrid_robust"],
        help="Threshold selection objective (default: hybrid_robust).",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="If constraints are unsatisfied, fall back to best unconstrained config.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    ensure_src_path(root)

    manifest_path = (root / args.manifest).resolve()
    cache_path = (root / args.cache).resolve()
    out_path = (root / args.out_json).resolve()

    # Prepare runtime env for stable CPU execution.
    os.environ["CFN_USE_EMBEDDINGS"] = os.getenv("CFN_USE_EMBEDDINGS", "true")
    os.environ["CFN_W2V2_MODEL"] = os.getenv("CFN_W2V2_MODEL", "WAV2VEC2_BASE")
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MPLCONFIGDIR"] = os.getenv("MPLCONFIGDIR", "/tmp/mpl")

    if args.build_manifest or not manifest_path.exists():
        rows = build_manifest(
            root=root,
            fakeav_root=(root / args.fakeav_root).resolve(),
            dfdc_root=(root / args.dfdc_root).resolve(),
            per_class_limit=args.per_class_per_dataset,
            seed=args.seed,
            out_path=manifest_path,
        )
        print(f"Built manifest: {manifest_path} ({len(rows)} rows)")
    else:
        rows = parse_manifest(manifest_path)
        print(f"Loaded manifest: {manifest_path} ({len(rows)} rows)")

    if not rows:
        raise RuntimeError("No rows found for evaluation.")

    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"Loaded cache: {cache_path} ({len(cache)} rows)")
    else:
        import src.cvi.api.inference_service as inf
        importlib.reload(inf)

        cache = []
        skipped = 0
        for i, row in enumerate(rows, start=1):
            p = Path(row.path)
            if not p.is_absolute():
                p = root / p
            try:
                out = inf.run_full_cvi_pipeline(str(p))
                frames = out.get("frames", [])
                probs = [float(f.get("fake_prob_smooth", f.get("fake_prob", 0.0))) for f in frames]
                mism = [float(f.get("av_mismatch", 0.0)) for f in frames]
                cache.append(
                    {
                        "path": str(p),
                        "label": int(row.label),
                        "dataset": row.dataset,
                        "probs": probs,
                        "mism": mism,
                    }
                )
            except Exception as exc:
                skipped += 1
                print(f"[skip] {p}: {exc}")
            if i % 20 == 0:
                print(f"Cached {i}/{len(rows)}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
        print(f"Saved cache: {cache_path} ({len(cache)} rows, skipped={skipped})")

    prob_grid = [float(x.strip()) for x in args.prob_grid.split(",") if x.strip()]
    ratio_grid = [float(x.strip()) for x in args.ratio_grid.split(",") if x.strip()]
    causal_grid = [float(x.strip()) for x in args.causal_grid.split(",") if x.strip()]
    rf_opts = [x.strip().lower() == "true" for x in args.require_flag_options.split(",") if x.strip()]

    best = None
    unconstrained_best = None
    for rf in rf_opts:
        for p in prob_grid:
            for r in ratio_grid:
                for c in causal_grid:
                    m = eval_combo(cache, p, r, c, rf)
                    overall = m["overall"]
                    per_dataset = m["per_dataset"]
                    if per_dataset:
                        uc_min_ds_rec = min(v["rec"] for v in per_dataset.values())
                        uc_min_ds_spec = min(v["spec"] for v in per_dataset.values())
                        uc_worst_ds_bal = min(v["bal_acc"] for v in per_dataset.values())
                    else:
                        uc_min_ds_rec = overall["rec"]
                        uc_min_ds_spec = overall["spec"]
                        uc_worst_ds_bal = overall["bal_acc"]

                    if args.selection_objective == "worst":
                        uc_key = (uc_worst_ds_bal, m["macro_bal_acc"], overall["bal_acc"], overall["f1"])
                    elif args.selection_objective == "macro":
                        uc_key = (m["macro_bal_acc"], m["macro_f1"], overall["bal_acc"], overall["f1"])
                    else:
                        uc_robust_score = 0.55 * uc_worst_ds_bal + 0.30 * m["macro_bal_acc"] + 0.15 * overall["f1"]
                        uc_key = (uc_robust_score, uc_worst_ds_bal, m["macro_bal_acc"], overall["f1"])

                    if unconstrained_best is None or uc_key > unconstrained_best["key"]:
                        unconstrained_best = {
                            "key": uc_key,
                            "require_flag": rf,
                            "prob": p,
                            "ratio": r,
                            "causal": c,
                            "metrics": m,
                            "min_domain_rec": uc_min_ds_rec,
                            "min_domain_spec": uc_min_ds_spec,
                            "worst_domain_bal_acc": uc_worst_ds_bal,
                        }

                    if overall["rec"] < args.min_rec or overall["spec"] < args.min_spec:
                        continue
                    if per_dataset:
                        min_ds_rec = min(v["rec"] for v in per_dataset.values())
                        min_ds_spec = min(v["spec"] for v in per_dataset.values())
                        worst_ds_bal = min(v["bal_acc"] for v in per_dataset.values())
                    else:
                        min_ds_rec = overall["rec"]
                        min_ds_spec = overall["spec"]
                        worst_ds_bal = overall["bal_acc"]

                    if min_ds_rec < args.min_domain_rec or min_ds_spec < args.min_domain_spec:
                        continue

                    if args.selection_objective == "worst":
                        key = (worst_ds_bal, m["macro_bal_acc"], overall["bal_acc"], overall["f1"])
                    elif args.selection_objective == "macro":
                        key = (m["macro_bal_acc"], m["macro_f1"], overall["bal_acc"], overall["f1"])
                    else:
                        robust_score = 0.55 * worst_ds_bal + 0.30 * m["macro_bal_acc"] + 0.15 * overall["f1"]
                        key = (robust_score, worst_ds_bal, m["macro_bal_acc"], overall["f1"])

                    if best is None or key > best["key"]:
                        best = {
                            "key": key,
                            "require_flag": rf,
                            "prob": p,
                            "ratio": r,
                            "causal": c,
                            "metrics": m,
                            "min_domain_rec": min_ds_rec,
                            "min_domain_spec": min_ds_spec,
                            "worst_domain_bal_acc": worst_ds_bal,
                        }

    used_fallback = False
    if best is None:
        if args.allow_fallback and unconstrained_best is not None:
            best = unconstrained_best
            used_fallback = True
            print("Warning: no threshold combo satisfied constraints; using best unconstrained config.")
        else:
            raise RuntimeError("No threshold combo satisfied constraints.")

    payload = {
        "manifest": str(manifest_path),
        "cache": str(cache_path),
        "best": best,
        "used_fallback": used_fallback,
        "recommend_env": {
            "CFN_PROB_THRESH": f"{best['prob']:.4f}",
            "CFN_RATIO_THRESH": f"{best['ratio']:.4f}",
            "CFN_CAUSAL_THRESH": f"{best['causal']:.4f}",
            "CFN_REQUIRE_FLAG": str(best["require_flag"]).lower(),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    overall = best["metrics"]["overall"]
    print("==== BEST MIXED CONFIG ====")
    print(
        f"MacroBalAcc={best['metrics']['macro_bal_acc']:.3f} MacroF1={best['metrics']['macro_f1']:.3f} "
        f"| Overall Acc={overall['acc']:.3f} BalAcc={overall['bal_acc']:.3f} "
        f"Rec={overall['rec']:.3f} Spec={overall['spec']:.3f}"
    )
    print(
        f"WorstDomainBalAcc={best['worst_domain_bal_acc']:.3f} "
        f"MinDomainRec={best['min_domain_rec']:.3f} MinDomainSpec={best['min_domain_spec']:.3f} "
        f"Objective={args.selection_objective}"
    )
    print(
        f"PROB={best['prob']:.2f} RATIO={best['ratio']:.2f} "
        f"CAUSAL={best['causal']:.2f} REQUIRE_FLAG={best['require_flag']}"
    )
    for ds, met in best["metrics"]["per_dataset"].items():
        print(
            f"[{ds}] Acc={met['acc']:.3f} BalAcc={met['bal_acc']:.3f} "
            f"Rec={met['rec']:.3f} Spec={met['spec']:.3f} F1={met['f1']:.3f}"
        )
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
