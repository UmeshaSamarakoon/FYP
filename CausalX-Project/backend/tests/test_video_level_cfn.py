from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cvi import video_level_cfn
from src.cvi.step46_precompute import STEP46_AV_COLUMNS, STEP46_PHYS_COLUMNS, read_step46_artifact
from src.modules.causal_fusion import CausalFusionNetworkV2


def _reset_video_level_caches() -> None:
    video_level_cfn._load_model.cache_clear()
    video_level_cfn._load_scaler.cache_clear()
    video_level_cfn._load_temporal_scorer.cache_clear()
    video_level_cfn._load_tabular_scorer.cache_clear()
    video_level_cfn._load_manifest_spec.cache_clear()
    video_level_cfn._load_selection_bundle.cache_clear()
    video_level_cfn._load_runtime_calibration.cache_clear()


def _write_model_dir(model_dir: Path, bias: float) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model = CausalFusionNetworkV2(
        av_dim=len(STEP46_AV_COLUMNS),
        phys_dim=len(STEP46_PHYS_COLUMNS),
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


def _zero_feature_map() -> dict[str, float]:
    return {name: 0.0 for name in [*STEP46_AV_COLUMNS, *STEP46_PHYS_COLUMNS]}


def _write_tabular_scorer(scorer_path: Path, *, constant_label: int) -> None:
    clf = DummyClassifier(strategy="constant", constant=int(constant_label))
    x = [[0.0] * (len(STEP46_AV_COLUMNS) + len(STEP46_PHYS_COLUMNS)) for _ in range(4)]
    y = [int(constant_label), int(constant_label), 1 - int(constant_label), 1 - int(constant_label)]
    clf.fit(x, y)
    joblib.dump(
        {
            "kind": "step46_tabular_scorer",
            "version": 1,
            "model_name": "dummy_constant",
            "feature_columns": [*STEP46_AV_COLUMNS, *STEP46_PHYS_COLUMNS],
            "threshold": 0.5,
            "model": clf,
        },
        scorer_path,
    )


def _write_temporal_scorer(scorer_path: Path, *, bias: float) -> None:
    model = video_level_cfn.RuntimeParityTemporalScorer(input_dim=4, channels=(8,), dropout=0.0)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.head[-1].bias.fill_(float(bias))
    torch.save(
        {
            "kind": "runtimeparity_temporal_scorer",
            "version": 1,
            "feature_columns": ["frame_presence_ratio", "lip_aperture_mean", "audio_rms_mean", "av_local_corr_mean"],
            "threshold": 0.5,
            "model_config": {"input_dim": 4, "channels": [8], "dropout": 0.0},
            "normalization_mean": [0.0, 0.0, 0.0, 0.0],
            "normalization_std": [1.0, 1.0, 1.0, 1.0],
            "model_state_dict": model.state_dict(),
        },
        scorer_path,
    )


def test_video_level_tabular_scorer_gate(tmp_path, monkeypatch):
    scorer_path = tmp_path / "tabular.joblib"
    _write_tabular_scorer(scorer_path, constant_label=1)
    _reset_video_level_caches()

    monkeypatch.setenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", str(scorer_path))
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_MODEL_DIR", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_PRECOMPUTE_DIR", str(tmp_path / "precompute"))
    monkeypatch.setattr(
        video_level_cfn,
        "extract_causal_features",
        lambda _path: _zero_feature_map(),
    )

    score = video_level_cfn.score_video_level_cfn(str(tmp_path / "clip.mp4"))

    assert score is not None
    assert score.decision_source == "video_level_tabular"
    assert score.model_mode == "tabular"
    assert score.video_fake == 1
    assert score.fake_prob >= 0.5


def test_video_level_default_tabular_is_disabled_when_env_is_unset(tmp_path, monkeypatch):
    scorer_path = tmp_path / "default_tabular.joblib"
    _write_tabular_scorer(scorer_path, constant_label=1)
    _reset_video_level_caches()

    monkeypatch.setattr(video_level_cfn, "_DEFAULT_TABULAR_SCORER_PATH", scorer_path)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", raising=False)

    assert video_level_cfn._resolve_tabular_scorer_path() is None


def test_video_level_temporal_scorer_gate(tmp_path, monkeypatch):
    scorer_path = tmp_path / "temporal.pt"
    _write_temporal_scorer(scorer_path, bias=2.0)
    _reset_video_level_caches()

    monkeypatch.setenv("CFN_VIDEO_LEVEL_TEMPORAL_SCORER_PATH", str(scorer_path))
    monkeypatch.delenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", "false")
    monkeypatch.delenv("CFN_VIDEO_LEVEL_MODEL_DIR", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", "false")
    monkeypatch.setattr(
        video_level_cfn,
        "write_runtimeparity_temporal_artifact",
        lambda _path: str((tmp_path / "artifact.json").resolve()),
    )
    monkeypatch.setattr(
        video_level_cfn,
        "read_runtimeparity_temporal_artifact",
        lambda _path: {"sequence": np.ones((8, 4), dtype=np.float32), "mask": np.ones(8, dtype=np.float32)},
    )

    score = video_level_cfn.score_video_level_cfn(str(tmp_path / "clip.mp4"))

    assert score is not None
    assert score.decision_source == "video_level_temporal"
    assert score.model_mode == "temporal"
    assert score.video_fake == 1
    assert score.fake_prob > 0.5


def test_video_level_cfn_single_model_gate(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _write_model_dir(model_dir, bias=2.0)
    _reset_video_level_caches()

    monkeypatch.setenv("CFN_VIDEO_LEVEL_MODEL_DIR", str(model_dir))
    monkeypatch.delenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", "false")
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_PRECOMPUTE_DIR", str(tmp_path / "precompute"))
    monkeypatch.setattr(
        video_level_cfn,
        "extract_causal_features",
        lambda _path: _zero_feature_map(),
    )

    score = video_level_cfn.score_video_level_cfn(str(tmp_path / "clip.mp4"))

    assert score is not None
    assert score.decision_source == "video_level_cfn_single"
    assert score.video_fake == 1
    assert score.fake_prob > 0.5
    assert score.artifact_csv_path is not None
    assert Path(score.artifact_csv_path).exists()


def test_video_level_cfn_default_manifest_is_used_when_present(tmp_path, monkeypatch):
    model_dir = tmp_path / "manifest-model"
    _write_model_dir(model_dir, bias=2.0)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "test-manifest",
                "artifacts": [
                    {
                        "fold": "fold_01",
                        "model_path": str(model_dir / "cfn_emb.pth"),
                        "scaler_path": None,
                    }
                ],
            }
        )
    )
    _reset_video_level_caches()

    monkeypatch.setattr(video_level_cfn, "_DEFAULT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(video_level_cfn, "_DEFAULT_RUNTIME_CALIBRATION_PATH", tmp_path / "missing_runtime_calibration.json")
    monkeypatch.delenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", "false")
    monkeypatch.delenv("CFN_VIDEO_LEVEL_MODEL_DIR", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_RUNTIME_CALIBRATION_JSON", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", "true")
    monkeypatch.setenv("CFN_VIDEO_LEVEL_PRECOMPUTE_DIR", str(tmp_path / "precompute"))
    monkeypatch.setattr(
        video_level_cfn,
        "extract_causal_features",
        lambda _path: _zero_feature_map(),
    )

    score = video_level_cfn.score_video_level_cfn(str(tmp_path / "clip.mp4"))

    assert score is not None
    assert score.decision_source == "video_level_cfn_ensemble"
    assert score.video_fake == 1
    assert score.fake_prob > 0.5


def test_video_level_cfn_can_disable_default_manifest(monkeypatch):
    _reset_video_level_caches()

    monkeypatch.delenv("CFN_VIDEO_LEVEL_MODEL_DIR", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", "false")
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", "false")

    def _should_not_extract(_path):
        raise AssertionError("feature extraction should not run when default manifest usage is disabled")

    monkeypatch.setattr(video_level_cfn, "extract_causal_features", _should_not_extract)

    score = video_level_cfn.score_video_level_cfn("clip.mp4")

    assert score is None


def test_video_level_default_manifest_is_disabled_when_env_is_unset(tmp_path, monkeypatch):
    model_dir = tmp_path / "manifest-model"
    _write_model_dir(model_dir, bias=2.0)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "test-manifest",
                "artifacts": [
                    {
                        "fold": "fold_01",
                        "model_path": str(model_dir / "cfn_emb.pth"),
                        "scaler_path": None,
                    }
                ],
            }
        )
    )
    _reset_video_level_caches()

    monkeypatch.setattr(video_level_cfn, "_DEFAULT_MANIFEST_PATH", manifest_path)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", raising=False)

    assert video_level_cfn._resolve_manifest_path() is None


def test_video_level_cfn_runtime_calibration_overrides_manifest_threshold(tmp_path, monkeypatch):
    model_dir = tmp_path / "manifest-model"
    _write_model_dir(model_dir, bias=2.0)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "test-manifest",
                "artifacts": [
                    {
                        "fold": "fold_01",
                        "model_path": str(model_dir / "cfn_emb.pth"),
                        "scaler_path": None,
                    }
                ],
            }
        )
    )
    calibration_path = tmp_path / "runtime_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "threshold_mode": "mean_prob",
                "ensemble_threshold": 0.95,
            }
        )
    )
    _reset_video_level_caches()

    monkeypatch.setattr(video_level_cfn, "_DEFAULT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(video_level_cfn, "_DEFAULT_RUNTIME_CALIBRATION_PATH", calibration_path)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", "false")
    monkeypatch.delenv("CFN_VIDEO_LEVEL_MODEL_DIR", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_RUNTIME_CALIBRATION_JSON", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", "true")
    monkeypatch.setenv("CFN_VIDEO_LEVEL_PRECOMPUTE_DIR", str(tmp_path / "precompute"))
    monkeypatch.setattr(
        video_level_cfn,
        "extract_causal_features",
        lambda _path: _zero_feature_map(),
    )

    score = video_level_cfn.score_video_level_cfn(str(tmp_path / "clip.mp4"))

    assert score is not None
    assert score.decision_source == "video_level_cfn_ensemble_calibrated"
    assert score.fake_prob > 0.5
    assert score.threshold == 0.95
    assert score.video_fake == 0


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
    monkeypatch.delenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", "false")
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_SELECTION_JSON", str(selection_path))
    monkeypatch.setenv("CFN_VIDEO_LEVEL_PRECOMPUTE_DIR", str(tmp_path / "precompute"))
    monkeypatch.setattr(
        video_level_cfn,
        "extract_causal_features",
        lambda _path: _zero_feature_map(),
    )

    score = video_level_cfn.score_video_level_cfn(str(tmp_path / "clip.mp4"))

    assert score is not None
    assert score.decision_source == "video_level_cfn_selection"
    assert score.video_fake == 1
    assert score.fake_prob > 0.5


def test_video_level_cfn_scores_saved_step46_artifact(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _write_model_dir(model_dir, bias=2.0)
    _reset_video_level_caches()

    monkeypatch.setenv("CFN_VIDEO_LEVEL_MODEL_DIR", str(model_dir))
    monkeypatch.delenv("CFN_VIDEO_LEVEL_TABULAR_SCORER_PATH", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_USE_DEFAULT_TABULAR", "false")
    monkeypatch.delenv("CFN_VIDEO_LEVEL_SELECTION_JSON", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_ENSEMBLE_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("CFN_VIDEO_LEVEL_USE_DEFAULT_MANIFEST", raising=False)
    monkeypatch.setenv("CFN_VIDEO_LEVEL_PRECOMPUTE_DIR", str(tmp_path / "precompute"))
    monkeypatch.setattr(video_level_cfn, "extract_causal_features", lambda _path: _zero_feature_map())

    artifact_csv_path = video_level_cfn.create_step46_precompute_artifact(
        str(tmp_path / "clip.mp4"),
        label=0,
        dataset="fakeavceleb",
        video_fake=0,
        audio_fake=0,
    )

    assert artifact_csv_path is not None
    artifact_row = read_step46_artifact(artifact_csv_path)
    assert artifact_row is not None
    assert artifact_row["dataset"] == "fakeavceleb"
    assert int(float(artifact_row["label"])) == 0
    assert Path(str(artifact_row["path"])).name == "clip.mp4"

    score = video_level_cfn.score_video_level_precomputed_csv(artifact_csv_path)

    assert score is not None
    assert score.decision_source == "video_level_cfn_single"
    assert score.artifact_csv_path == str(Path(artifact_csv_path).resolve())
