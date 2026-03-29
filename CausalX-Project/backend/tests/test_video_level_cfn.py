from __future__ import annotations

import json
from pathlib import Path

import torch

from src.cvi.cfn_frame_inference import _AV_FEATURE_ORDER, _PHYS_FEATURE_ORDER
from src.cvi import video_level_cfn
from src.modules.causal_fusion import CausalFusionNetworkV2


def _reset_video_level_caches() -> None:
    video_level_cfn._load_model.cache_clear()
    video_level_cfn._load_scaler.cache_clear()
    video_level_cfn._load_manifest_spec.cache_clear()
    video_level_cfn._load_selection_bundle.cache_clear()


def _write_model_dir(model_dir: Path, bias: float) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model = CausalFusionNetworkV2(
        av_dim=len(_AV_FEATURE_ORDER),
        phys_dim=len(_PHYS_FEATURE_ORDER),
        enable_av_input_layernorm=True,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.classifier[0].bias.fill_(bias)
    torch.save(model.state_dict(), model_dir / "cfn_emb.pth")
    (model_dir / "cfn_threshold_report.json").write_text(
        json.dumps({"chosen_epoch_report": {"selection_threshold": 0.5}})
    )
    (model_dir / "cfn_temperature.json").write_text(json.dumps({"temperature": 1.0}))


def test_video_level_cfn_single_model_gate(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _write_model_dir(model_dir, bias=2.0)
    _reset_video_level_caches()

    monkeypatch.setenv("CFN_VIDEO_LEVEL_MODEL_DIR", str(model_dir))
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", raising=False)
    monkeypatch.setattr(
        video_level_cfn,
        "extract_causal_features",
        lambda _path: {name: 0.0 for name in [*_AV_FEATURE_ORDER, *_PHYS_FEATURE_ORDER]},
    )

    score = video_level_cfn.score_video_level_cfn(str(tmp_path / "clip.mp4"))

    assert score is not None
    assert score.decision_source == "video_level_cfn_single"
    assert score.video_fake == 1
    assert score.fake_prob > 0.5


def test_video_level_cfn_requires_explicit_artifact_config(monkeypatch):
    _reset_video_level_caches()

    monkeypatch.delenv("CFN_VIDEO_LEVEL_MODEL_DIR", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", raising=False)

    def _should_not_extract(_path):
        raise AssertionError("feature extraction should not run without an explicit video-level artifact")

    monkeypatch.setattr(video_level_cfn, "extract_causal_features", _should_not_extract)

    score = video_level_cfn.score_video_level_cfn("clip.mp4")

    assert score is None


def test_video_level_cfn_selection_json_gate(tmp_path, monkeypatch):
    model_dir = tmp_path / "selection-model"
    _write_model_dir(model_dir, bias=2.0)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "best_model": {
                    "model_dir": str(model_dir),
                    "clean_threshold": 0.5,
                }
            }
        )
    )
    _reset_video_level_caches()

    monkeypatch.delenv("CFN_VIDEO_LEVEL_MODEL_DIR", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_SELECTION_JSON", str(selection_path))
    monkeypatch.setattr(
        video_level_cfn,
        "extract_causal_features",
        lambda _path: {name: 0.0 for name in [*_AV_FEATURE_ORDER, *_PHYS_FEATURE_ORDER]},
    )

    score = video_level_cfn.score_video_level_cfn(str(tmp_path / "clip.mp4"))

    assert score is not None
    assert score.decision_source == "video_level_cfn_selection"
    assert score.video_fake == 1
    assert score.fake_prob > 0.5
