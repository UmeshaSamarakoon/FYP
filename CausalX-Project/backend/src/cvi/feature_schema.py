from __future__ import annotations

from typing import Iterable


BASELINE_AV_FEATURES = [
    "lip_variance",
    "av_correlation",
    "av_lag_frames",
]


EXTENDED_AV_FEATURES = [
    "lip_mean",
    "lip_std",
    "lip_range",
    "lip_velocity_mean",
    "lip_velocity_std",
    "audio_rms_mean",
    "audio_rms_std",
    "av_corr_05_mean",
    "av_corr_05_std",
    "av_corr_10_mean",
    "av_corr_10_std",
    "av_corr_20_mean",
    "av_corr_20_std",
    "av_peak_corr",
    "av_peak_lag_sec",
    "av_peak_prominence",
    "av_onset_corr",
]


BASELINE_PHYS_FEATURES = [
    "jitter_mean",
    "jitter_std",
]


EXTENDED_PHYS_FEATURES = [
    "mouth_flow_mean",
    "mouth_flow_std",
    "mouth_aspect_mean",
    "mouth_aspect_std",
    "mouth_area_mean",
    "mouth_area_std",
    "mouth_area_delta_std",
    "mouth_asym_mean",
    "mouth_asym_std",
    "det_count",
]


EMBEDDING_AV_FEATURES = [
    "tcn_visual_emb",
    "wav2vec_audio_emb",
]


def _ordered_present(candidates: list[str], present: Iterable[str]) -> list[str]:
    present_set = set(present)
    return [c for c in candidates if c in present_set]


def resolve_feature_columns(
    available_columns: Iterable[str],
    use_embeddings: bool,
    profile: str = "auto",
) -> tuple[list[str], list[str]]:
    """
    Resolve AV/physical feature columns.

    profile:
      - baseline: baseline-only columns
      - extended: baseline + extended columns (missing columns should be zero-filled upstream)
      - auto: baseline + any extended columns present in input data
    """
    profile = (profile or "auto").strip().lower()
    if profile not in {"baseline", "extended", "auto"}:
        raise ValueError(f"Unsupported feature profile: {profile}")

    available = list(available_columns)
    if profile == "baseline":
        av_cols = list(BASELINE_AV_FEATURES)
        phys_cols = list(BASELINE_PHYS_FEATURES)
    elif profile == "extended":
        av_cols = list(BASELINE_AV_FEATURES) + list(EXTENDED_AV_FEATURES)
        phys_cols = list(BASELINE_PHYS_FEATURES) + list(EXTENDED_PHYS_FEATURES)
    else:
        av_cols = list(BASELINE_AV_FEATURES) + _ordered_present(EXTENDED_AV_FEATURES, available)
        phys_cols = list(BASELINE_PHYS_FEATURES) + _ordered_present(EXTENDED_PHYS_FEATURES, available)

    if use_embeddings:
        # Keep embedding columns deterministic in order; zero-fill if absent upstream.
        av_cols.extend(EMBEDDING_AV_FEATURES)

    return av_cols, phys_cols

