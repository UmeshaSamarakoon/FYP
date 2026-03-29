from __future__ import annotations

import json
from typing import Any

from src.cvi.storage.logs_store import log_event


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_frame_metric(frames: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for frame in frames or []:
        value = _safe_float(frame.get(key))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def _mean_bool_frame_metric(frames: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for frame in frames or []:
        value = frame.get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
    if not values:
        return None
    return float(sum(values) / len(values))


def build_hidden_score_summary(pipeline_output: dict[str, Any]) -> dict[str, Any]:
    frames = list(pipeline_output.get("frames") or [])
    public_causal_breach = _safe_float(pipeline_output.get("causal_breach_score"))
    raw_causal_breach = _mean_frame_metric(frames, "causal_breach_score")

    summary: dict[str, Any] = {
        "video_label": "FAKE" if pipeline_output.get("video_fake") else "REAL",
        "hidden_overall_score": _safe_float(pipeline_output.get("overall_score")),
        "hidden_decision_source": pipeline_output.get("decision_source"),
        "hidden_legacy_fake_ratio": _safe_float(pipeline_output.get("legacy_fake_ratio")),
        "hidden_calibrator_score": _safe_float(pipeline_output.get("calibrator_score")),
        "hidden_scm_enabled": bool(pipeline_output.get("scm_enabled", False)),
        "public_causal_breach_score": public_causal_breach,
        "raw_frame_mean_causal_breach_score": raw_causal_breach,
        "raw_frame_mean_fake_prob": _mean_frame_metric(frames, "fake_prob"),
        "raw_frame_mean_fake_prob_smooth": _mean_frame_metric(frames, "fake_prob_smooth"),
        "raw_frame_mean_av_mismatch": _mean_frame_metric(frames, "av_mismatch"),
        "raw_frame_mean_scm_z": _mean_frame_metric(frames, "scm_z"),
        "raw_frame_scm_violation_ratio": _mean_bool_frame_metric(frames, "scm_violation"),
        "public_highlight_count": len(pipeline_output.get("highlight_timestamps") or []),
        "public_causal_segment_count": len(pipeline_output.get("causal_segments") or []),
    }

    # Keep terminal output compact by dropping empty diagnostics.
    return {k: v for k, v in summary.items() if v is not None}


def emit_hidden_score_summary(analysis_id: str, video_name: str, pipeline_output: dict[str, Any]) -> dict[str, Any]:
    summary = build_hidden_score_summary(pipeline_output)
    print(
        f"[CVI hidden scores] analysis_id={analysis_id} video={video_name} "
        f"{json.dumps(summary, sort_keys=True)}",
        flush=True,
    )
    log_event(analysis_id, "hidden_scores_reported", summary)
    return summary
