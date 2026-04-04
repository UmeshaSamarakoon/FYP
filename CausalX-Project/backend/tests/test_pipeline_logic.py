import os
import asyncio

from src.cvi.api import main
from src.cvi.api.main import _safe_upload_path
from src.cvi import pipeline
from src.cvi.pipeline import CausalInferenceEngine, summarize_video
from src.cvi.video_calibrator_features import FEATURE_NAMES, build_video_feature_vector


def test_safe_upload_path_strips_traversal_and_is_unique():
    fname = "../evil/clip.mp4"
    path1 = _safe_upload_path(fname)
    path2 = _safe_upload_path(fname)

    # Paths should live under uploads/ and differ due to UUID prefix
    assert path1.startswith("uploads" + os.sep)
    assert path2.startswith("uploads" + os.sep)
    assert os.path.basename(path1).endswith("clip.mp4")
    assert os.path.basename(path2).endswith("clip.mp4")
    assert path1 != path2
    assert ".." not in path1 and ".." not in path2


def test_summarize_video_uses_causal_flags():
    frames = [
        {"timestamp": 0.0, "fake_prob": 0.2, "causal_break": True},
        {"timestamp": 0.1, "fake_prob": 0.1, "causal_break": True},
        {"timestamp": 0.2, "fake_prob": 0.1, "causal_break": False},
        {"timestamp": 0.3, "fake_prob": 0.1, "causal_break": False},
    ]

    # With prob threshold 0.6 these would be REAL, but causal flags should flip
    video_fake, confidence, highlights = summarize_video(
        frames,
        prob_thresh=0.6,
        ratio_thresh=0.3,
        prob_key="fake_prob",
        flag_key="causal_break",
    )

    assert video_fake == 1
    assert confidence == 0.5  # two of four frames flagged via causal_break
    assert highlights == [0.0, 0.1]


def test_health_check_reports_runtime_pipeline(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_runtime_pipeline_summary",
        lambda: {
            "decision_pipeline": "live_frame_pipeline",
            "video_level_mode": "disabled",
            "video_level_default_tabular_enabled": False,
            "video_level_default_manifest_enabled": False,
        },
    )

    payload = asyncio.run(main.health_check())

    assert payload["status"] == "ok"
    assert payload["decision_pipeline"] == "live_frame_pipeline"


def test_video_calibrator_feature_vector_matches_feature_names():
    frames = [
        {"timestamp": 0.0, "fake_prob_smooth": 0.2, "av_mismatch": 0.1},
        {"timestamp": 0.1, "fake_prob_smooth": 0.7, "av_mismatch": 0.8},
        {"timestamp": 0.2, "fake_prob_smooth": 0.6, "av_mismatch": 0.9},
        {"timestamp": 0.3, "fake_prob_smooth": 0.3, "av_mismatch": 0.2},
    ]

    vector = build_video_feature_vector(frames, prob_key="fake_prob_smooth")

    assert vector.shape == (len(FEATURE_NAMES),)
    assert len(FEATURE_NAMES) == 27


def test_engine_uses_threshold_rule_without_video_level_overrides(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "run_cfn_on_video",
        lambda *_args, **_kwargs: [
            {"timestamp": 0.0, "fake_prob": 0.1, "av_mismatch": 0.1},
            {"timestamp": 0.1, "fake_prob": 0.2, "av_mismatch": 0.1},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "get_last_run_diagnostics",
        lambda: {"used_facemesh_fallback": False, "audio_backends": ["librosa"]},
    )
    monkeypatch.setattr(pipeline, "score_video_level_cfn", lambda _path: None)

    engine = CausalInferenceEngine(
        prob_thresh=0.6,
        ratio_thresh=0.8,
        smooth_window=1,
        chunk_seconds=10,
        causal_thresh=0.6,
        max_seconds=30.0,
        require_flag=True,
        min_segment_frames=3,
    )

    output = engine.run("clip.mp4")

    assert output["decision_source"] == "threshold_rule"
    assert output["runtime_diagnostics"]["audio_backends"] == ["librosa"]
