from __future__ import annotations

from dataclasses import dataclass

from src.cvi.frame_causal_extractor import (
    extract_frame_level_features,
    compute_av_mismatch,
    get_video_meta,
)
from src.cvi.cfn_frame_inference import run_cfn_on_video
from src.cvi.api.inference_service import (
    add_causal_breaks,
    build_segments,
    overall_video_score,
    smooth_fake_probs,
    summarize_video,
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

    def run(self, video_path: str):
        frame_results = run_cfn_on_video(
            video_path,
            threshold=self.prob_thresh,
            causal_threshold=self.causal_thresh,
            chunk_seconds=self.chunk_seconds,
            max_seconds=self.max_seconds,
        )

        frame_results, prob_key = smooth_fake_probs(frame_results, self.smooth_window)
        frame_results = add_causal_breaks(frame_results, causal_thresh=self.causal_thresh)
        causal_segments = build_segments(frame_results, flag_key="causal_break")

        video_fake, confidence, highlight_times = summarize_video(
            frame_results,
            prob_thresh=self.prob_thresh,
            ratio_thresh=self.ratio_thresh,
            prob_key=prob_key,
        )

        overall_score = overall_video_score(frame_results, prob_key=prob_key)

        return {
            "video_fake": video_fake,
            "fake_confidence": confidence,
            "overall_score": overall_score,
            "highlight_timestamps": highlight_times,
            "causal_segments": causal_segments,
            "frames": frame_results,
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
