#!/usr/bin/env python3
"""
Mine false-positive and false-negative video paths from cached frame outputs.

This script uses the same decision rule as mixed_domain_sweep.py for a single
threshold configuration and writes:
  - false positives (real predicted fake)
  - false negatives (fake predicted real)
  - combined mistake list (path-only, usable for hard-negative upweighting)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def decide(row, prob_t: float, ratio_t: float, causal_t: float, require_flag: bool) -> int:
    probs = np.array(row.get("probs", []) or [], dtype=float)
    mism = np.array(row.get("mism", []) or [], dtype=float)
    if probs.size == 0:
        return 0
    if require_flag:
        suspicious = (probs >= prob_t) & (mism >= causal_t)
    else:
        suspicious = (probs >= prob_t) | (mism >= causal_t)
    return int(float(np.mean(suspicious)) >= ratio_t)


def main():
    parser = argparse.ArgumentParser(description="Mine FP/FN paths from cache and thresholds.")
    parser.add_argument("--cache", required=True, help="Cache JSON (eval_cache_*.json)")
    parser.add_argument("--prob", type=float, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--causal", type=float, required=True)
    parser.add_argument("--require-flag", type=str, default="false")
    parser.add_argument("--out-fp", default="data/processed/error_fp.tsv")
    parser.add_argument("--out-fn", default="data/processed/error_fn.tsv")
    parser.add_argument("--out-combined", default="data/processed/error_combined_paths.tsv")
    args = parser.parse_args()

    require_flag = parse_bool(args.require_flag)
    rows = json.loads(Path(args.cache).read_text())

    fp, fn = [], []
    for row in rows:
        y = int(row.get("label", 0))
        pred = decide(row, args.prob, args.ratio, args.causal, require_flag)
        path = str(row.get("path", "")).strip()
        if not path:
            continue
        if pred == 1 and y == 0:
            fp.append(path)
        elif pred == 0 and y == 1:
            fn.append(path)

    out_fp = Path(args.out_fp)
    out_fn = Path(args.out_fn)
    out_comb = Path(args.out_combined)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    out_fp.write_text("\n".join(fp) + ("\n" if fp else ""))
    out_fn.write_text("\n".join(fn) + ("\n" if fn else ""))

    combined = sorted(set(fp + fn))
    out_comb.write_text("\n".join(combined) + ("\n" if combined else ""))

    print(f"FP: {len(fp)} -> {out_fp}")
    print(f"FN: {len(fn)} -> {out_fn}")
    print(f"Combined unique: {len(combined)} -> {out_comb}")


if __name__ == "__main__":
    main()

