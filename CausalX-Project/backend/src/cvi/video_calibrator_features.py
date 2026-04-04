from __future__ import annotations

import math

import numpy as np


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
    "prob_top5_mean",
    "mism_top5_mean",
    "prob_last_quarter_delta",
    "mism_last_quarter_delta",
    "longest_prob_ge_0_55_run_ratio",
    "longest_prob_ge_0_70_run_ratio",
    "longest_mism_ge_0_70_run_ratio",
    "longest_joint_run_ratio",
    "joint_prob55_mism70_ratio",
    "prob55_transition_rate",
    "mism70_transition_rate",
    "joint_transition_rate",
]


def _stats(x: np.ndarray) -> tuple[float, float, float, float, float]:
    return (
        float(np.mean(x)),
        float(np.std(x)),
        float(np.percentile(x, 90)),
        float(np.percentile(x, 95)),
        float(np.max(x)),
    )


def _longest_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask:
        if bool(value):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _transition_rate(mask: np.ndarray) -> float:
    if len(mask) < 2:
        return 0.0
    return float(np.mean(mask[1:] != mask[:-1]))


def _topk_mean(x: np.ndarray, k: int) -> float:
    if len(x) == 0:
        return 0.0
    k = min(int(k), len(x))
    return float(np.mean(np.partition(x, len(x) - k)[-k:]))


def build_video_feature_vector(frames, prob_key: str = "fake_prob") -> np.ndarray:
    """
    Build a compact video-level feature vector from frame-level signals.
    The extra temporal features stay within the existing frame pipeline and
    only summarize patterns already present in the per-frame outputs.
    """
    if not frames:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    probs = np.array([f.get(prob_key, 0.0) for f in frames], dtype=np.float32)
    mism = np.array([f.get("av_mismatch", 0.0) for f in frames], dtype=np.float32)
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    mism = np.nan_to_num(mism, nan=0.0, posinf=1.0, neginf=0.0)

    p_mean, p_std, p_p90, p_p95, p_max = _stats(probs)
    m_mean, m_std, m_p90, m_p95, m_max = _stats(mism)

    if len(probs) > 1 and p_std > 1e-8 and m_std > 1e-8:
        corr = float(np.mean((probs - p_mean) * (mism - m_mean)) / (p_std * m_std))
        if not math.isfinite(corr):
            corr = 0.0
    else:
        corr = 0.0

    ratio_p70 = float(np.mean(probs >= 0.70))
    ratio_p80 = float(np.mean(probs >= 0.80))
    ratio_m70 = float(np.mean(mism >= 0.70))
    ratio_m80 = float(np.mean(mism >= 0.80))

    prob55 = probs >= 0.55
    prob70 = probs >= 0.70
    mism70 = mism >= 0.70
    joint = prob55 & mism70
    quarter = max(1, len(probs) // 4)

    return np.array(
        [
            p_mean,
            p_std,
            p_p90,
            p_p95,
            p_max,
            m_mean,
            m_std,
            m_p90,
            m_p95,
            m_max,
            corr,
            ratio_p70,
            ratio_p80,
            ratio_m70,
            ratio_m80,
            _topk_mean(probs, 5),
            _topk_mean(mism, 5),
            float(np.mean(probs[-quarter:]) - np.mean(probs[:quarter])),
            float(np.mean(mism[-quarter:]) - np.mean(mism[:quarter])),
            float(_longest_run(prob55) / len(probs)),
            float(_longest_run(prob70) / len(probs)),
            float(_longest_run(mism70) / len(probs)),
            float(_longest_run(joint) / len(probs)),
            float(np.mean(joint)),
            _transition_rate(prob55),
            _transition_rate(mism70),
            _transition_rate(joint),
        ],
        dtype=np.float32,
    )
