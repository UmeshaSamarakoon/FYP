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
    # Optional upgraded visual/audio embeddings.
    "effnet_b4_face_emb",
    "lip_roi_emb",
    "wav2vec2_base_ft_emb",
]


NEXTGEN_AV_FEATURES = [
    # Cross-modal confidence / lag stability features.
    "av_sync_gap",
    "av_lag_abs",
    "av_peak_lag_abs_sec",
    "av_confidence_proxy",
    # Cross-modal embedding discrepancy features.
    "emb_gap_tcn_w2v2",
    "emb_gap_effnet_w2v2",
    "emb_gap_lip_w2v2",
    "emb_gap_tcn_effnet",
    "emb_gap_tcn_lip",
    "emb_gap_effnet_lip",
    # Optional CLIP-style proxy embeddings for semantic adaptation.
    "clip_face_proxy_emb",
    "clip_lip_proxy_emb",
    "clip_audio_proxy_emb",
]


NEXTGEN_AV_OPTIONAL_FEATURES = [
    # Optional true CLIP columns (used if present in CSV).
    "clip_face_emb",
    "clip_lip_emb",
    "clip_audio_emb",
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
    # Video-artifact engineered features (compression/noise/dropout proxies).
    "video_motion_noise_ratio",
    "video_shape_noise_ratio",
    "video_temporal_instability",
    "video_detection_dropout",
    "video_compression_proxy",
]


NEXTGEN_PHYS_FEATURES = [
    # Distortion/discrepancy aggregates (D3-style signals).
    "artifact_breach_index",
    "artifact_stability_index",
    "artifact_quality_index",
    "av_artifact_interaction",
    "motion_shape_coupling",
]


EMBEDDING_AV_FEATURES = [
    "tcn_visual_emb",
    "wav2vec_audio_emb",
    "effnet_b4_face_emb",
    "lip_roi_emb",
    "wav2vec2_base_ft_emb",
]


LIP_STREAM_FEATURES = [
    # Dedicated mouth/Lip-ROI motion stream for 3-way fusion.
    "lip_mean",
    "lip_std",
    "lip_range",
    "lip_velocity_mean",
    "lip_velocity_std",
    "mouth_aspect_mean",
    "mouth_aspect_std",
    "mouth_area_mean",
    "mouth_area_std",
    "mouth_area_delta_std",
    "mouth_asym_mean",
    "mouth_asym_std",
    "mouth_flow_mean",
    "mouth_flow_std",
    "det_count",
    # Optional learned lip ROI embedding.
    "lip_roi_emb",
]


def _ordered_present(candidates: list[str], present: Iterable[str]) -> list[str]:
    present_set = set(present)
    return [c for c in candidates if c in present_set]


def _dedupe_keep_order(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


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
    if profile not in {"baseline", "extended", "nextgen", "auto"}:
        raise ValueError(f"Unsupported feature profile: {profile}")

    available = list(available_columns)
    if profile == "baseline":
        av_cols = list(BASELINE_AV_FEATURES)
        phys_cols = list(BASELINE_PHYS_FEATURES)
    elif profile == "extended":
        av_cols = list(BASELINE_AV_FEATURES) + list(EXTENDED_AV_FEATURES)
        phys_cols = list(BASELINE_PHYS_FEATURES) + list(EXTENDED_PHYS_FEATURES)
    elif profile == "nextgen":
        av_cols = (
            list(BASELINE_AV_FEATURES)
            + list(EXTENDED_AV_FEATURES)
            + list(NEXTGEN_AV_FEATURES)
            + _ordered_present(NEXTGEN_AV_OPTIONAL_FEATURES, available)
        )
        phys_cols = list(BASELINE_PHYS_FEATURES) + list(EXTENDED_PHYS_FEATURES) + list(NEXTGEN_PHYS_FEATURES)
    else:
        av_cols = (
            list(BASELINE_AV_FEATURES)
            + _ordered_present(EXTENDED_AV_FEATURES, available)
            + _ordered_present(NEXTGEN_AV_FEATURES, available)
            + _ordered_present(NEXTGEN_AV_OPTIONAL_FEATURES, available)
        )
        phys_cols = (
            list(BASELINE_PHYS_FEATURES)
            + _ordered_present(EXTENDED_PHYS_FEATURES, available)
            + _ordered_present(NEXTGEN_PHYS_FEATURES, available)
        )

    if use_embeddings:
        # Keep embedding columns deterministic in order; zero-fill if absent upstream.
        av_cols.extend(EMBEDDING_AV_FEATURES)

    return _dedupe_keep_order(av_cols), _dedupe_keep_order(phys_cols)
