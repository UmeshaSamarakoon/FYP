import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cvi.api.result_reporting import (
    build_hidden_score_summary,
    emit_hidden_score_summary,
)


def test_hidden_score_summary_keeps_raw_causal_breach_for_real_video():
    pipeline_output = {
        "video_fake": 0,
        "overall_score": 0.18,
        "causal_breach_score": 0.0,
        "decision_source": "threshold_rule",
        "legacy_fake_ratio": 0.18,
        "calibrator_score": None,
        "scm_enabled": False,
        "highlight_timestamps": [],
        "causal_segments": [],
        "frames": [
            {
                "causal_breach_score": 0.31,
                "fake_prob": 0.25,
                "fake_prob_smooth": 0.24,
                "av_mismatch": 0.16,
                "scm_z": 0.75,
                "scm_violation": False,
            },
            {
                "causal_breach_score": 0.49,
                "fake_prob": 0.35,
                "fake_prob_smooth": 0.34,
                "av_mismatch": 0.26,
                "scm_z": 1.25,
                "scm_violation": True,
            },
            {
                "causal_breach_score": 0.20,
                "fake_prob": 0.18,
                "fake_prob_smooth": 0.19,
                "av_mismatch": 0.11,
                "scm_z": 0.55,
                "scm_violation": False,
            },
        ],
    }

    summary = build_hidden_score_summary(pipeline_output)

    assert summary["video_label"] == "REAL"
    assert summary["public_causal_breach_score"] == 0.0
    assert summary["raw_frame_mean_causal_breach_score"] == (0.31 + 0.49 + 0.20) / 3
    assert summary["raw_frame_mean_fake_prob"] == (0.25 + 0.35 + 0.18) / 3
    assert summary["raw_frame_mean_fake_prob_smooth"] == (0.24 + 0.34 + 0.19) / 3
    assert summary["raw_frame_mean_av_mismatch"] == (0.16 + 0.26 + 0.11) / 3
    assert summary["raw_frame_mean_scm_z"] == (0.75 + 1.25 + 0.55) / 3
    assert summary["raw_frame_scm_violation_ratio"] == 1 / 3
    assert summary["hidden_overall_score"] == 0.18
    assert summary["hidden_legacy_fake_ratio"] == 0.18


def test_emit_hidden_score_summary_prints_hidden_backend_metrics(monkeypatch, capsys):
    pipeline_output = {
        "video_fake": 0,
        "overall_score": 0.18,
        "causal_breach_score": 0.0,
        "decision_source": "video_calibrator",
        "legacy_fake_ratio": 0.18,
        "calibrator_score": 0.27,
        "scm_enabled": True,
        "highlight_timestamps": [],
        "causal_segments": [],
        "frames": [
            {
                "causal_breach_score": 0.31,
                "fake_prob": 0.25,
                "fake_prob_smooth": 0.24,
                "av_mismatch": 0.16,
                "scm_z": 0.75,
                "scm_violation": False,
            },
            {
                "causal_breach_score": 0.49,
                "fake_prob": 0.35,
                "fake_prob_smooth": 0.34,
                "av_mismatch": 0.26,
                "scm_z": 1.25,
                "scm_violation": True,
            },
            {
                "causal_breach_score": 0.20,
                "fake_prob": 0.18,
                "fake_prob_smooth": 0.19,
                "av_mismatch": 0.11,
                "scm_z": 0.55,
                "scm_violation": False,
            },
        ],
    }
    logged_events = []

    monkeypatch.setattr(
        "src.cvi.api.result_reporting.log_event",
        lambda analysis_id, event, metadata=None: logged_events.append(
            (analysis_id, event, metadata)
        ),
    )

    summary = emit_hidden_score_summary("analysis-1", "real_clip.mp4", pipeline_output)

    stdout = capsys.readouterr().out.strip()
    assert stdout.startswith(
        "[CVI hidden scores] analysis_id=analysis-1 video=real_clip.mp4 "
    )
    payload = json.loads(stdout[stdout.index("{"):])
    assert payload == summary
    assert payload["video_label"] == "REAL"
    assert payload["raw_frame_mean_causal_breach_score"] == (0.31 + 0.49 + 0.20) / 3
    assert payload["raw_frame_mean_fake_prob"] == (0.25 + 0.35 + 0.18) / 3
    assert payload["raw_frame_mean_fake_prob_smooth"] == (0.24 + 0.34 + 0.19) / 3
    assert payload["raw_frame_mean_av_mismatch"] == (0.16 + 0.26 + 0.11) / 3
    assert payload["raw_frame_mean_scm_z"] == (0.75 + 1.25 + 0.55) / 3
    assert payload["raw_frame_scm_violation_ratio"] == 1 / 3
    assert payload["hidden_overall_score"] == 0.18
    assert payload["hidden_legacy_fake_ratio"] == 0.18
    assert payload["hidden_calibrator_score"] == 0.27
    assert payload["hidden_decision_source"] == "video_calibrator"
    assert payload["hidden_scm_enabled"] is True
    assert logged_events == [
        ("analysis-1", "hidden_scores_reported", summary),
    ]
