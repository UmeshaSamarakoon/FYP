import os
import numpy as np

from src.modules.embeddings import visual_tcn_embedding, wav2vec_embedding


class FeatureExtractor:
    """
    Embedding helpers for visual TCN and wav2vec2.
    Falls back to zeros if checkpoints/models are missing.
    """

    def __init__(self):
        self.visual_ckpt = os.getenv("CFN_VISUAL_TCN_PATH")
        self.wav2vec_model = os.getenv("CFN_W2V2_MODEL", "WAV2VEC2_BASE")

    def get_visual_embeddings(self, lip_signal: np.ndarray):
        return visual_tcn_embedding(lip_signal, checkpoint=self.visual_ckpt)

    def get_audio_embeddings(self, waveform: np.ndarray, sr: int):
        return wav2vec_embedding(waveform, sr, model_name=self.wav2vec_model)
