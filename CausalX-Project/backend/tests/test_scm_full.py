import numpy as np

from src.cvi.scm import StructuralCausalModel, run_scm


def test_structural_causal_model_intervention_shapes():
    rng = np.random.default_rng(7)
    n = 50
    jitter = rng.normal(0, 1, n).astype(np.float32)
    lips = (0.8 * jitter + 0.1 + rng.normal(0, 0.02, n)).astype(np.float32)
    audio = (1.2 * lips + 0.4 * jitter + 0.05 + rng.normal(0, 0.02, n)).astype(np.float32)

    scm = StructuralCausalModel().fit(lips=lips, audio=audio, jitter=jitter)
    lip_pred, audio_pred = scm.predict(lips=lips, jitter=jitter)

    assert lip_pred.shape == lips.shape
    assert audio_pred.shape == audio.shape

    do_audio = scm.intervention_do_lip(do_lip=np.zeros_like(lips), jitter=jitter)
    assert do_audio.shape == audio.shape



def test_run_scm_enriches_frames_with_full_outputs():
    frames = []
    for i in range(10):
        t = i * 0.04
        jitter = float(np.sin(i / 3))
        lip = 0.5 * jitter + 0.2
        audio = 1.1 * lip + 0.3 * jitter + 0.05
        frames.append({"timestamp": t, "jitter": jitter, "lip_aperture": lip, "audio_rms": audio})

    out = run_scm(frames, z_threshold=2.0)
    sample = out[0]

    for key in [
        "scm_graph",
        "scm_params",
        "scm_lip_pred",
        "scm_audio_pred",
        "scm_lip_resid",
        "scm_resid",
        "scm_lip_z",
        "scm_audio_z",
        "scm_z",
        "scm_violation",
        "scm_do_audio_static_lip",
    ]:
        assert key in sample
