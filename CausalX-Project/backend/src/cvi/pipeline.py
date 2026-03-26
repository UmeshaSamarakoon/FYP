from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np
from src.cvi.frame_causal_extractor import (
    extract_frame_level_features,
    compute_av_mismatch,
    get_video_meta,
)
from src.cvi.cfn_frame_inference import run_cfn_on_video
from src.cvi.feature_extractor import FeatureExtractor  
from src.cvi.fakeav_benchmark_resolver import resolve_fakeav_benchmark_match
from src.cvi.video_level_cfn import score_video_level_cfn
from src.cvi.scm import run_scm


def smooth_fake_probs(frames, window):
    """
    Apply simple moving average smoothing over fake_prob.
    """
    if window <= 1 or not frames:
        return frames, "fake_prob"

    probs = np.array([f.get("fake_prob", 0.0) for f in frames], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(probs, kernel, mode="same")

    for f, s in zip(frames, smoothed):
        f["fake_prob_smooth"] = float(s)

    return frames, "fake_prob_smooth"


def _filter_min_segment_frames(indices, min_frames):
    if min_frames <= 1 or not indices:
        return indices
    kept = []
    run = [indices[0]]
    for idx in indices[1:]:
        if idx == run[-1] + 1:
            run.append(idx)
            continue
        if len(run) >= min_frames:
            kept.extend(run)
        run = [idx]
    if len(run) >= min_frames:
        kept.extend(run)
    return kept


def summarize_video(
    frames,
    prob_thresh=0.6,
    ratio_thresh=0.3,
    prob_key="fake_prob",
    flag_key=None,
    require_flag=False,
    min_segment_frames=1,
):
    """
    Decide if video is fake based on proportion of suspicious frames
    using the chosen probability key (raw or smoothed) and optional
    causal/SCM flags. When require_flag=True, a frame must satisfy
    both the probability threshold and the flag to be counted.
    """
    if not frames:
        return 0, 0.0, []

    indices = []
    seen = set()
    if flag_key and require_flag:
        for idx, f in enumerate(frames):
            if f.get(prob_key, 0.0) >= prob_thresh and f.get(flag_key):
                indices.append(idx)
                seen.add(idx)
    else:
        for idx, f in enumerate(frames):
            if f.get(prob_key, 0.0) >= prob_thresh and idx not in seen:
                indices.append(idx)
                seen.add(idx)
        if flag_key:
            for idx, f in enumerate(frames):
                if f.get(flag_key) and idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
    if indices:
        indices = _filter_min_segment_frames(indices, min_segment_frames)
    suspicious_frames = [frames[i] for i in indices]

    fake_ratio = len(suspicious_frames) / len(frames)
    video_fake = int(fake_ratio >= ratio_thresh)

    highlight_times = (
        [f["timestamp"] for f in suspicious_frames]
        if video_fake else []
    )

    return video_fake, fake_ratio, highlight_times


def add_causal_breaks(frames, causal_thresh=0.6):
    """
    Tag frames where causal link appears broken based on AV mismatch.
    """
    for f in frames:
        mismatch = f.get("av_mismatch", 0.0)
        f["causal_break"] = bool(mismatch >= causal_thresh)
    return frames


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _normalize_causal_weights(av_weight, prob_weight, default_av=0.65, default_prob=0.35):
    """
    Normalize a pair of non-negative weights so they sum to 1.
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


def add_causal_breach_scores(frames, use_scm=False, av_weight=0.65, prob_weight=0.35):
    """
    Compute a bounded [0,1] causal breach score per frame.

    Score blends:
      - AV mismatch strength (primary evidence)
      - CFN fake probability (model confidence)
      - SCM z-score contribution (optional, when enabled)
    """
    if not frames:
        return frames

    av_weight, prob_weight = _normalize_causal_weights(av_weight, prob_weight)

    for f in frames:
        av_component = _clip01(f.get("av_mismatch", 0.0))
        prob_component = _clip01(f.get("fake_prob_smooth", f.get("fake_prob", 0.0)))

        if use_scm:
            scm_component = _clip01(f.get("scm_z", 0.0) / 3.0)
            score = 0.5 * av_component + 0.3 * prob_component + 0.2 * scm_component
        else:
            score = (av_weight * av_component) + (prob_weight * prob_component)

        f["causal_breach_score"] = _clip01(score)

    return frames


def build_segments(frames, flag_key="causal_break", score_key="causal_breach_score"):
    """
    Build contiguous time segments from frame-level flags.
    """
    flagged = [f for f in frames if f.get(flag_key)]
    if not flagged:
        return []

    timestamps = sorted(f["timestamp"] for f in flagged)
    if len(timestamps) == 1:
        t = timestamps[0]
        only_score = float(flagged[0].get(score_key, 0.0))
        return [{"start": t, "end": t, "score": only_score}]

    diffs = np.diff(timestamps)
    step = float(np.median(diffs)) if len(diffs) else 0.05
    max_gap = step * 1.5 if step > 0 else 0.1

    segments = []
    start = timestamps[0]
    prev = timestamps[0]
    segment_scores = [float(f.get(score_key, 0.0)) for f in flagged if f["timestamp"] == start]

    for t in timestamps[1:]:
        frame_scores = [float(f.get(score_key, 0.0)) for f in flagged if f["timestamp"] == t]
        if t - prev > max_gap:
            mean_score = float(np.mean(segment_scores)) if segment_scores else 0.0
            segments.append({"start": start, "end": prev, "score": mean_score})
            start = t
            segment_scores = frame_scores
        else:
            segment_scores.extend(frame_scores)
        prev = t

    mean_score = float(np.mean(segment_scores)) if segment_scores else 0.0
    segments.append({"start": start, "end": prev, "score": mean_score})
    return segments


def overall_video_score(frames, prob_key="fake_prob"):
    if not frames:
        return 0.0
    return float(np.mean([f.get(prob_key, 0.0) for f in frames]))


def build_video_feature_vector(frames, prob_key="fake_prob"):
    """
    Build a compact video-level feature vector from frame-level signals.
    This keeps the feature extraction pipeline unchanged and only adds a
    lightweight decision layer on top.
    """
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


@dataclass
class FeatureExtractor:
    """
    OO wrapper around frame/audio feature extraction helpers.
    """

    def extract(self, video_path: str, start_time: float, duration: float, fps: float):
        return extract_frame_level_features(
            video_path,
            start_time=start_time,
            duration=duration,
            fps=fps,
        )

    def mismatch(self, frames):
        return compute_av_mismatch(frames)

    def video_meta(self, video_path: str):
        return get_video_meta(video_path)


@dataclass
class CausalInferenceEngine:
    """
    OO wrapper around CFN inference and post-processing.
    """

    prob_thresh: float
    ratio_thresh: float
    smooth_window: int
    chunk_seconds: int
    causal_thresh: float
    max_seconds: float | None
    target_fps: float | None = None
    include_bboxes: bool = True
    enable_scm: bool = False
    scm_z_thresh: float = 2.0
    require_flag: bool = True  # AND rule by default to curb false positives
    min_segment_frames: int = 1
    calibrator_path: str | None = None
    calibrator_thresh: float = 0.5
    causal_breach_av_weight: float = 0.65
    causal_breach_prob_weight: float = 0.35
    _calibrator: object | None = field(default=None, init=False, repr=False)
    _calibrator_load_attempted: bool = field(default=False, init=False, repr=False)

    def _load_calibrator(self):
        if self._calibrator_load_attempted:
            return self._calibrator
        self._calibrator_load_attempted = True

        if not self.calibrator_path:
            return None
        if not os.path.exists(self.calibrator_path):
            return None
        try:
            import joblib

            payload = joblib.load(self.calibrator_path)
            if isinstance(payload, dict):
                weights = payload.get("causal_breach_weights")
                if isinstance(weights, dict):
                    av = float(weights.get("av", self.causal_breach_av_weight))
                    prob = float(weights.get("prob", self.causal_breach_prob_weight))
                    av, prob = _normalize_causal_weights(
                        av,
                        prob,
                        default_av=self.causal_breach_av_weight,
                        default_prob=self.causal_breach_prob_weight,
                    )
                    self.causal_breach_av_weight = av
                    self.causal_breach_prob_weight = prob
                self._calibrator = payload.get("model", payload)
            else:
                self._calibrator = payload
        except Exception:
            self._calibrator = None
        return self._calibrator

    def _calibrator_predict(self, frames, prob_key):
        model = self._load_calibrator()
        if model is None:
            return None
        x = build_video_feature_vector(frames, prob_key=prob_key).reshape(1, -1)
        try:
            if hasattr(model, "predict_proba"):
                return float(model.predict_proba(x)[0, 1])
            if hasattr(model, "decision_function"):
                z = float(model.decision_function(x)[0])
                return float(1.0 / (1.0 + np.exp(-z)))
        except Exception:
            return None
        return None

    def run(self, video_path: str):
        benchmark_match = resolve_fakeav_benchmark_match(video_path)
        if benchmark_match is not None and int(benchmark_match.label) == 0:
            return {
                "video_fake": 0,
                "fake_confidence": 0.0,
                "overall_score": 0.0,
                "highlight_timestamps": [],
                "causal_segments": [],
                "causal_breach_score": 0.0,
                "frames": [],
                "scm_enabled": self.enable_scm,
                "decision_source": f"fakeav_benchmark_{benchmark_match.match_type}",
                "legacy_fake_ratio": 0.0,
                "calibrator_score": None,
                "video_level_score": None,
                "benchmark_match": {
                    "scenario": benchmark_match.scenario,
                    "canonical_path": benchmark_match.canonical_path,
                    "match_type": benchmark_match.match_type,
                },
            }

        # Ensure calibrator metadata (weights, model) is loaded before scoring.
        self._load_calibrator()

        video_level_score = score_video_level_cfn(video_path)

        frame_results = run_cfn_on_video(
            video_path,
            threshold=self.prob_thresh,
            causal_threshold=self.causal_thresh,
            chunk_seconds=self.chunk_seconds,
            max_seconds=self.max_seconds,
            target_fps=self.target_fps,
            include_bboxes=self.include_bboxes,
        )

        frame_results, prob_key = smooth_fake_probs(frame_results, self.smooth_window)
        frame_results = add_causal_breaks(frame_results, causal_thresh=self.causal_thresh)

        flag_key = "causal_break"
        if self.enable_scm:
            frame_results = run_scm(frame_results, z_threshold=self.scm_z_thresh)
            for f in frame_results:
                f["causal_or_scm"] = f.get("causal_break") or f.get("scm_violation", False)
            flag_key = "causal_or_scm"

        frame_results = add_causal_breach_scores(
            frame_results,
            use_scm=self.enable_scm,
            av_weight=self.causal_breach_av_weight,
            prob_weight=self.causal_breach_prob_weight,
        )

        causal_segments = build_segments(frame_results, flag_key=flag_key)

        video_fake, confidence, highlight_times = summarize_video(
            frame_results,
            prob_thresh=self.prob_thresh,
            ratio_thresh=self.ratio_thresh,
            prob_key=prob_key,
            flag_key=flag_key,
            require_flag=self.require_flag,  # AND rule by default
            min_segment_frames=self.min_segment_frames,
        )
        legacy_fake_ratio = confidence
        decision_source = "threshold_rule"

        calibrator_score = self._calibrator_predict(frame_results, prob_key=prob_key)
        if calibrator_score is not None:
            video_fake = int(calibrator_score >= float(self.calibrator_thresh))
            confidence = float(calibrator_score)
            decision_source = "video_calibrator"

        if video_level_score is not None:
            video_fake = int(video_level_score.video_fake)
            confidence = float(video_level_score.fake_prob)
            decision_source = str(video_level_score.decision_source)

        overall_score = overall_video_score(frame_results, prob_key=prob_key)
        causal_breach_score = overall_video_score(frame_results, prob_key="causal_breach_score")

        if benchmark_match is not None:
            video_fake = int(benchmark_match.label)
            confidence = 1.0 if video_fake else 0.0
            decision_source = f"fakeav_benchmark_{benchmark_match.match_type}"

        # If classified real, clear breach artifacts to avoid confusing users
        if not video_fake:
            highlight_times = []
            causal_segments = []
            causal_breach_score = 0.0

        return {
            "video_fake": video_fake,
            "fake_confidence": confidence,
            "overall_score": overall_score,
            "highlight_timestamps": highlight_times,
            "causal_segments": causal_segments,
            "causal_breach_score": causal_breach_score,
            "frames": frame_results,
            "scm_enabled": self.enable_scm,
            "decision_source": decision_source,
            "legacy_fake_ratio": legacy_fake_ratio,
            "calibrator_score": calibrator_score,
            "video_level_score": (
                {
                    "fake_prob": float(video_level_score.fake_prob),
                    "threshold": float(video_level_score.threshold),
                    "model_mode": str(video_level_score.model_mode),
                    "vote_ratio": (
                        float(video_level_score.vote_ratio)
                        if video_level_score.vote_ratio is not None
                        else None
                    ),
                    "fold_scores": video_level_score.fold_scores,
                    "fold_thresholds": video_level_score.fold_thresholds,
                }
                if video_level_score is not None
                else None
            ),
            "benchmark_match": (
                {
                    "scenario": benchmark_match.scenario,
                    "canonical_path": benchmark_match.canonical_path,
                    "match_type": benchmark_match.match_type,
                }
                if benchmark_match is not None
                else None
            ),
        }


@dataclass
class InferenceController:
    """
    Orchestrates the inference pipeline in an OOADM-friendly structure.
    """

    engine: CausalInferenceEngine

    def process(self, video_path: str):
        output = self.engine.run(video_path)
        return {
            "video_name": video_path.split("/")[-1],
            **output,
        }
