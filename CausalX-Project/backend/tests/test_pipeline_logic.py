import os

from src.cvi.api.main import _safe_upload_path
from src.cvi.pipeline import summarize_video


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
