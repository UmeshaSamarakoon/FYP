import numpy as np


def _fit_linear(y, X):
    """Return beta solving min ||Xb - y||^2 with bias."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta  # last term is bias


def _zscore(v):
    v = np.asarray(v, dtype=np.float32)
    return (v - v.mean()) / (v.std() + 1e-6)


def run_scm(frames, z_threshold: float = 2.0):
    """
    Simple structural causal model:
      lip_aperture  --->  audio_rms
                 \\       /
                 jitter (physical) is treated as a potential confounder

    We fit audio_rms ≈ a1*lip + a2*jitter + b.
    Residual z-scores flag causal violations (audio not explained by lip+phys).
    """
    if not frames or len(frames) < 3:
        return frames

    lips = np.array([f.get("lip_aperture", 0.0) for f in frames], dtype=np.float32)
    audio = np.array([f.get("audio_rms", 0.0) for f in frames], dtype=np.float32)
    jitter = np.array([f.get("jitter", 0.0) for f in frames], dtype=np.float32)

    try:
        # X = [lip, jitter]
        X = np.stack([lips, jitter], axis=1)
        beta = _fit_linear(audio, X)
        w_lip, w_jit, bias = beta
        audio_pred = w_lip * lips + w_jit * jitter + bias
        resid = audio - audio_pred
        z = _zscore(resid)
    except Exception:
        return frames

    for f, r, zval, ap, jp, pred in zip(frames, resid, z, audio, jitter, audio_pred):
        f["scm_audio_pred"] = float(pred)
        f["scm_resid"] = float(r)
        f["scm_z"] = float(abs(zval))
        f["scm_violation"] = bool(abs(zval) >= z_threshold)
        # optional counterfactual: remove lip effect
        f["scm_cf_audio_no_lip"] = float(w_jit * jp + bias)

    return frames
