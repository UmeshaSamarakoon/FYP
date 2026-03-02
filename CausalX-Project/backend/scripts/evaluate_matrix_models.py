#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_manifest(path: Path):
    rows = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if "\t" not in line:
            continue
        path_str, label_str = line.rsplit("\t", 1)
        rows.append((path_str, int(label_str)))
    return rows


def eval_combo(cache, prob_t, ratio_t, causal_t, require_flag):
    tp = tn = fp = fn = 0
    for item in cache:
        y = item["label"]
        probs = item["probs"]
        mism = item["mism"]
        if probs.size == 0:
            pred = 0
        else:
            if require_flag:
                suspicious = (probs >= prob_t) & (mism >= causal_t)
            else:
                suspicious = (probs >= prob_t) | (mism >= causal_t)
            pred = int(float(np.mean(suspicious)) >= ratio_t)

        if pred == 1 and y == 1:
            tp += 1
        elif pred == 0 and y == 0:
            tn += 1
        elif pred == 1 and y == 0:
            fp += 1
        else:
            fn += 1

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate matrix checkpoints with threshold sweep.")
    parser.add_argument("--manifest", default="eval_manifest_balanced.tsv")
    parser.add_argument("--runs", default="A,B,C,D")
    parser.add_argument("--model-dir", default="models/matrix_runs")
    parser.add_argument("--visual-tcn-path", default="models/visual_tcn.pth")
    parser.add_argument("--out-json", default="models/matrix_runs/sweep_results.json")
    parser.add_argument("--prob-grid", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--ratio-grid", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--causal-grid", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    args = parser.parse_args()

    root = Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    manifest = (root / args.manifest).resolve()
    rows = parse_manifest(manifest)
    if not rows:
        raise RuntimeError(f"No valid rows in manifest: {manifest}")

    run_ids = [x.strip() for x in args.runs.split(",") if x.strip()]
    model_dir = (root / args.model_dir).resolve()
    out_json = (root / args.out_json).resolve()

    prob_grid = [float(x) for x in args.prob_grid.split(",")]
    ratio_grid = [float(x) for x in args.ratio_grid.split(",")]
    causal_grid = [float(x) for x in args.causal_grid.split(",")]

    os.environ["PYTHONPATH"] = "."
    os.environ["CFN_USE_EMBEDDINGS"] = "true"
    os.environ["CFN_W2V2_MODEL"] = os.getenv("CFN_W2V2_MODEL", "WAV2VEC2_BASE")
    os.environ["CFN_VISUAL_TCN_PATH"] = str((root / args.visual_tcn_path).resolve())
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MPLCONFIGDIR"] = os.getenv("MPLCONFIGDIR", "/tmp/mpl")
    os.environ["CFN_SMOOTH_WINDOW"] = os.getenv("CFN_SMOOTH_WINDOW", "5")

    all_results = {}
    best_global = None

    for run_id in run_ids:
        model_path = model_dir / f"cfn_emb_{run_id}.pth"
        scaler_src = model_dir / f"cfn_scaler_{run_id}.pkl"
        scaler_dst = root / "models" / "cfn_scaler.pkl"
        if not model_path.exists() or not scaler_src.exists():
            print(f"[skip] missing artifacts for run {run_id}")
            continue

        os.environ["CFN_EMB_MODEL_PATH"] = str(model_path)
        shutil.copy2(scaler_src, scaler_dst)

        # Reset lazy-loaded artifacts between runs.
        import src.cvi.cfn_frame_inference as fi
        fi._model = None
        fi._scaler = None
        fi._AV_DIM = None

        import src.cvi.api.inference_service as inf
        importlib.reload(inf)

        print(f"[run {run_id}] caching frame outputs for {len(rows)} videos")
        cache = []
        skipped = 0
        for idx, (path_str, label) in enumerate(rows, start=1):
            p = Path(path_str)
            if not p.is_absolute():
                p = root / p
            try:
                out = inf.run_full_cvi_pipeline(str(p))
                frames = out.get("frames", [])
                probs = np.array([f.get("fake_prob_smooth", f.get("fake_prob", 0.0)) for f in frames], dtype=float)
                mism = np.array([f.get("av_mismatch", 0.0) for f in frames], dtype=float)
                cache.append({"label": int(label), "probs": probs, "mism": mism})
            except Exception as exc:
                skipped += 1
                print(f"[run {run_id}] skip {p}: {exc}")
            if idx % 25 == 0:
                print(f"[run {run_id}] processed {idx}/{len(rows)}")

        best = None
        for rf in (False, True):
            for p in prob_grid:
                for r in ratio_grid:
                    for c in causal_grid:
                        metrics = eval_combo(cache, p, r, c, rf)
                        key = (metrics["bal_acc"], metrics["f1"], metrics["acc"])
                        if best is None or key > best["key"]:
                            best = {
                                "key": key,
                                "require_flag": rf,
                                "prob": p,
                                "ratio": r,
                                "causal": c,
                                "metrics": metrics,
                            }

        all_results[run_id] = {
            "model_path": str(model_path),
            "scaler_path": str(scaler_src),
            "skipped": skipped,
            "best": best,
        }
        m = best["metrics"]
        print(
            f"[run {run_id}] best BalAcc={m['bal_acc']:.3f} "
            f"Acc={m['acc']:.3f} F1={m['f1']:.3f} "
            f"Rec={m['rec']:.3f} Spec={m['spec']:.3f} "
            f"at PROB={best['prob']:.2f} RATIO={best['ratio']:.2f} "
            f"CAUSAL={best['causal']:.2f} REQUIRE_FLAG={best['require_flag']}"
        )

        if best_global is None or best["key"] > best_global["key"]:
            best_global = {"run_id": run_id, **best}

    payload = {
        "manifest": str(manifest),
        "results": all_results,
        "best_global": best_global,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {out_json}")

    if best_global:
        m = best_global["metrics"]
        print("==== BEST GLOBAL ====")
        print(
            f"RUN={best_global['run_id']} BalAcc={m['bal_acc']:.3f} "
            f"Acc={m['acc']:.3f} F1={m['f1']:.3f} Rec={m['rec']:.3f} Spec={m['spec']:.3f}"
        )
        print(
            f"PROB={best_global['prob']:.2f} RATIO={best_global['ratio']:.2f} "
            f"CAUSAL={best_global['causal']:.2f} REQUIRE_FLAG={best_global['require_flag']}"
        )


if __name__ == "__main__":
    main()
