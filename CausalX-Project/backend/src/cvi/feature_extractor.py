import numpy as np
import torch


class FeatureExtractor:
    """
    Lightweight embedding helpers (placeholder for future models).
    """

    def get_visual_embeddings(self, lip_signal: np.ndarray):
        # Placeholder: return zero embeddings; replace with real model if available.
        return np.zeros((len(lip_signal), 1), dtype=np.float32)

    def get_audio_embeddings(self, waveform: np.ndarray, sr: int):
        # Placeholder: return zero embeddings; replace with real model if available.
        return np.zeros((len(waveform), 1), dtype=np.float32)
