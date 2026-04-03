import os
import sys
import importlib
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from src.cvi.api.inference_service import build_inference_controller
from src.cvi.api import inference_service
from src.cvi import cfn_frame_inference


SAMPLE_VIDEO = PROJECT_ROOT / "data/validation_evaluation_videos/evaluation/data/raw/fakeavceleb/FakeVideo-FakeAudio/African/women/id00220/00027_id00220_wavtolip.mp4"


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Sample FakeAVCeleb video is not available in this workspace.")
def test_inference_controller_processes_sample_video():
    controller = build_inference_controller()
    output = controller.process(str(SAMPLE_VIDEO))

    assert isinstance(output, dict)
    assert isinstance(output.get("frames"), list)
    assert len(output["frames"]) > 0
    assert "fake_confidence" in output
    assert "overall_score" in output
    assert "causal_segments" in output
    frame_probs = np.asarray([float(frame.get("fake_prob", 0.0)) for frame in output["frames"]], dtype=np.float32)
    assert float(frame_probs.std()) > 0.0


def test_prob_threshold_uses_safe_floor_when_inferred_threshold_is_too_low(monkeypatch):
    monkeypatch.delenv("CFN_PROB_THRESH", raising=False)
    monkeypatch.setattr(inference_service, "resolve_default_probability_threshold", lambda: 0.17667756484283345)

    assert inference_service._resolve_prob_thresh() == pytest.approx(0.6)


def test_prob_threshold_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("CFN_PROB_THRESH", "0.73")
    monkeypatch.setattr(inference_service, "resolve_default_probability_threshold", lambda: 0.17667756484283345)

    assert inference_service._resolve_prob_thresh() == pytest.approx(0.73)


def test_frame_inference_defaults_to_single_checkpoint_when_manifest_not_explicit(monkeypatch):
    monkeypatch.delenv("CFN_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_EMB_MODEL_PATH", raising=False)

    assert cfn_frame_inference._resolve_manifest_path(single_model_override="") is None
    assert cfn_frame_inference._resolve_model_paths_for_threshold() == [cfn_frame_inference._DEFAULT_EMB_MODEL_PATH]


def test_require_flag_defaults_to_true(monkeypatch):
    monkeypatch.delenv("CFN_REQUIRE_FLAG", raising=False)
    reloaded = importlib.reload(inference_service)
    try:
        assert reloaded.REQUIRE_FLAG is True
    finally:
        importlib.reload(inference_service)
