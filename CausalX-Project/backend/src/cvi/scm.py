from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-6


def _fit_linear(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return beta solving min ||Xb - y||^2 with bias."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta.astype(np.float32)


def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return (v - v.mean()) / (v.std() + EPS)


@dataclass
class StructuralCausalModel:
    """
    Full linear SCM with explicit structural equations and intervention/counterfactual support.

    Graph:
      jitter ---------> lip_aperture ---------> audio_rms
         |--------------------------->|

    Structural equations:
      lip_aperture = a_jitter * jitter + b_lip + u_lip
      audio_rms    = c_lip * lip_aperture + c_jitter * jitter + b_audio + u_audio
    """

    lip_beta: np.ndarray | None = None
    audio_beta: np.ndarray | None = None

    def fit(self, lips: np.ndarray, audio: np.ndarray, jitter: np.ndarray) -> "StructuralCausalModel":
        jitter_col = jitter.reshape(-1, 1)

        # lip <- jitter
        self.lip_beta = _fit_linear(lips, jitter_col)  # [a_jitter, b_lip]

        # audio <- lip, jitter
        X_audio = np.stack([lips, jitter], axis=1)
        self.audio_beta = _fit_linear(audio, X_audio)  # [c_lip, c_jitter, b_audio]
        return self

    def _check_fitted(self) -> None:
        if self.lip_beta is None or self.audio_beta is None:
            raise ValueError("SCM must be fitted before inference")

    def predict(self, lips: np.ndarray, jitter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        a_jit, b_lip = self.lip_beta
        c_lip, c_jit, b_audio = self.audio_beta

        lip_pred = a_jit * jitter + b_lip
        audio_pred = c_lip * lips + c_jit * jitter + b_audio
        return lip_pred.astype(np.float32), audio_pred.astype(np.float32)

    def intervention_do_lip(self, do_lip: np.ndarray, jitter: np.ndarray) -> np.ndarray:
        """Intervention: do(lip_aperture = do_lip) and predict resulting audio."""
        self._check_fitted()
        c_lip, c_jit, b_audio = self.audio_beta
        return (c_lip * do_lip + c_jit * jitter + b_audio).astype(np.float32)

def run_scm(frames, z_threshold: float = 2.0):
    """
    Run explicit SCM fit + anomaly detection + intervention metrics.
    """
    if not frames or len(frames) < 3:
        return frames

    lips = np.array([f.get("lip_aperture", 0.0) for f in frames], dtype=np.float32)
    audio = np.array([f.get("audio_rms", 0.0) for f in frames], dtype=np.float32)
    jitter = np.array([f.get("jitter", 0.0) for f in frames], dtype=np.float32)

    try:
        scm = StructuralCausalModel().fit(lips=lips, audio=audio, jitter=jitter)

        lip_pred, audio_pred = scm.predict(lips=lips, jitter=jitter)
        lip_resid = lips - lip_pred
        audio_resid = audio - audio_pred

        lip_z = np.abs(_zscore(lip_resid))
        audio_z = np.abs(_zscore(audio_resid))
        joint_z = np.maximum(lip_z, audio_z)

        # Interventional baseline: no lip motion
        do_lip_static = np.zeros_like(lips, dtype=np.float32)
        do_audio_static_lip = scm.intervention_do_lip(do_lip=do_lip_static, jitter=jitter)

    except Exception:
        return frames

    a_jit, b_lip = scm.lip_beta
    c_lip, c_jit, b_audio = scm.audio_beta

    for i, f in enumerate(frames):
        f["scm_graph"] = "jitter->lip,jitter->audio,lip->audio"
        f["scm_params"] = {
            "lip_from_jitter": float(a_jit),
            "lip_bias": float(b_lip),
            "audio_from_lip": float(c_lip),
            "audio_from_jitter": float(c_jit),
            "audio_bias": float(b_audio),
        }

        f["scm_lip_pred"] = float(lip_pred[i])
        f["scm_audio_pred"] = float(audio_pred[i])
        f["scm_lip_resid"] = float(lip_resid[i])
        f["scm_resid"] = float(audio_resid[i])

        f["scm_lip_z"] = float(lip_z[i])
        f["scm_audio_z"] = float(audio_z[i])
        f["scm_z"] = float(joint_z[i])
        f["scm_violation"] = bool(joint_z[i] >= z_threshold)

        f["scm_do_audio_static_lip"] = float(do_audio_static_lip[i])

    return frames
