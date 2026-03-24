import os
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

from src.cvi.api.inference_service import build_inference_controller


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
