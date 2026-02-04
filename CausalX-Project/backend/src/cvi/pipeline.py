from __future__ import annotations

from dataclasses import dataclass, field

from src.cvi.frame_causal_extractor import (
    extract_frame_level_features,
    compute_av_mismatch,
    get_video_meta,
)
import numpy as np
import torch
import torchaudio

from src.cvi.cfn_frame_inference import run_cfn_on_video
from src.modules.temporal_conv import TemporalConvNet


@dataclass
class SCMChecker:
    """
    Lightweight SCM-style dependency check over AV mismatch signals.
    """

    def check_av_dependency(self, frames) -> float:
        if not frames:
            return 0.0
        mismatches = np.array([f.get("av_mismatch", 0.0) for f in frames], dtype=np.float32)
        score = float(1.0 - np.clip(mismatches.mean(), 0.0, 1.0))
        return score


@dataclass
class FeatureExtractor:
    """
    OO wrapper around frame/audio feature extraction helpers.
    """

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _tcn: TemporalConvNet | None = field(default=None, init=False)
    _wav2vec_bundle: torchaudio.pipelines.Wav2Vec2Bundle | None = field(default=None, init=False)
    _wav2vec_model: torch.nn.Module | None = field(default=None, init=False)

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

    def _init_tcn(self) -> None:
        if self._tcn is None:
            self._tcn = TemporalConvNet(in_channels=1, channels=[16, 32, 64]).to(self.device)
            self._tcn.eval()

    def _init_wav2vec(self) -> None:
        if self._wav2vec_model is None:
            self._wav2vec_bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self._wav2vec_model = self._wav2vec_bundle.get_model().to(self.device)
            self._wav2vec_model.eval()

    def get_visual_embeddings(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute visual embeddings using a lightweight TCN over a 1D signal.
        Expects frames shaped (T,) or (T, 1). Returns pooled embedding.
        """
        self._init_tcn()
        tensor = torch.tensor(frames, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(1)
        tensor = tensor.transpose(0, 1).unsqueeze(0)  # (B=1, C=1, T)
        with torch.no_grad():
            embedding = self._tcn(tensor).squeeze(0).cpu().numpy()
        return embedding

    def get_audio_embeddings(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute audio embeddings using Wav2Vec2.
        """
        self._init_wav2vec()
        tensor = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            features, _ = self._wav2vec_model(tensor)
        return features.mean(dim=1).squeeze(0).cpu().numpy()


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
    enable_scm_checks: bool = False
    scm_checker: SCMChecker = field(default_factory=SCMChecker)

    @staticmethod
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

    @staticmethod
    def summarize_video(frames, prob_thresh=0.6, ratio_thresh=0.3, prob_key="fake_prob"):
        """
        Decide if video is fake based on proportion of suspicious frames
        using the chosen probability key (raw or smoothed).
        """
        if not frames:
            return 0, 0.0, []

        suspicious_frames = [
            f for f in frames if f.get(prob_key, 0.0) >= prob_thresh
        ]

        fake_ratio = len(suspicious_frames) / len(frames)
        video_fake = int(fake_ratio >= ratio_thresh)

        highlight_times = (
            [f["timestamp"] for f in suspicious_frames]
            if video_fake else []
        )

        return video_fake, fake_ratio, highlight_times

    @staticmethod
    def add_causal_breaks(frames, causal_thresh=0.6):
        """
        Tag frames where causal link appears broken based on AV mismatch.
        """
        for f in frames:
            mismatch = f.get("av_mismatch", 0.0)
            f["causal_break"] = bool(mismatch >= causal_thresh)
        return frames

    @staticmethod
    def build_segments(frames, flag_key="causal_break"):
        """
        Build contiguous time segments from frame-level flags.
        """
        flagged = [f for f in frames if f.get(flag_key)]
        if not flagged:
            return []

        timestamps = sorted(f["timestamp"] for f in flagged)
        if len(timestamps) == 1:
            t = timestamps[0]
            return [[t, t]]

        diffs = np.diff(timestamps)
        step = float(np.median(diffs)) if len(diffs) else 0.05
        max_gap = step * 1.5 if step > 0 else 0.1

        segments = []
        start = timestamps[0]
        prev = timestamps[0]

        for t in timestamps[1:]:
            if t - prev > max_gap:
                segments.append([start, prev])
                start = t
            prev = t

        segments.append([start, prev])
        return segments

    @staticmethod
    def overall_video_score(frames, prob_key="fake_prob"):
        if not frames:
            return 0.0
        return float(np.mean([f.get(prob_key, 0.0) for f in frames]))

    def run(self, video_path: str):
        frame_results = run_cfn_on_video(
            video_path,
            threshold=self.prob_thresh,
            causal_threshold=self.causal_thresh,
            chunk_seconds=self.chunk_seconds,
            max_seconds=self.max_seconds,
        )

        frame_results, prob_key = self.smooth_fake_probs(frame_results, self.smooth_window)
        frame_results = self.add_causal_breaks(frame_results, causal_thresh=self.causal_thresh)
        causal_segments = self.build_segments(frame_results, flag_key="causal_break")

        video_fake, confidence, highlight_times = self.summarize_video(
            frame_results,
            prob_thresh=self.prob_thresh,
            ratio_thresh=self.ratio_thresh,
            prob_key=prob_key,
        )

        overall_score = self.overall_video_score(frame_results, prob_key=prob_key)

        response = {
            "video_fake": video_fake,
            "fake_confidence": confidence,
            "overall_score": overall_score,
            "highlight_timestamps": highlight_times,
            "causal_segments": causal_segments,
            "frames": frame_results,
        }

        if self.enable_scm_checks:
            response["scm_dependency_score"] = self.scm_checker.check_av_dependency(frame_results)

        return response


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
