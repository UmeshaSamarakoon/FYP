import os
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import torchaudio
except Exception:  # noqa: BLE001
    torchaudio = None  # Allow graceful degradation when torchaudio is missing

from src.modules.temporal_conv import TemporalConvNet


class VisualTCN(nn.Module):
    """
    Lightweight temporal conv over lip aperture sequence to produce a
    fixed-size visual embedding. Intended to be trained separately and
    loaded from a checkpoint.
    """

    def __init__(self, in_channels: int = 1, channels: list[int] | None = None, out_dim: int = 64):
        super().__init__()
        channels = channels or [16, 32, 64]
        self.tcn = TemporalConvNet(in_channels=in_channels, channels=channels, kernel_size=3)
        self.head = nn.Linear(channels[-1], out_dim)

    def forward(self, lip_seq: torch.Tensor) -> torch.Tensor:
        # lip_seq: (B, T) -> reshape to (B, 1, T)
        x = lip_seq.unsqueeze(1)
        feats = self.tcn(x)  # (B, C)
        return self.head(feats)  # (B, out_dim)


class AudioWav2VecEmbedder:
    """
    Wrapper around torchaudio wav2vec2 pipelines. Falls back to zeros if
    torchaudio or the model is unavailable.
    """

    def __init__(self, model_name: str = "WAV2VEC2_BASE"):
        if torchaudio is None:
            raise RuntimeError("torchaudio is not available for wav2vec2 embeddings")
        try:
            bundle = getattr(torchaudio.pipelines, model_name)
        except AttributeError as exc:
            raise RuntimeError(f"torchaudio pipeline {model_name} not found") from exc

        self.model = bundle.get_model()
        self.model.eval()
        self.sample_rate = bundle.sample_rate

    def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        if waveform.size == 0:
            return np.zeros((1, 1), dtype=np.float32)

        # Resample if needed
        if sr != self.sample_rate:
            if torchaudio is None:
                return np.zeros((1, 1), dtype=np.float32)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
            wav = resampler(wav)
        else:
            wav = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

        # wav2vec2 feature extractor expects at least a few samples; pad if too short
        min_len = 10  # conservative lower bound; conv1d kernels are <= 10
        if wav.shape[-1] < min_len:
            pad = min_len - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))

        with torch.no_grad():
            out, _ = self.model(wav)
            # Mean pool over time
            emb = out.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            return emb.reshape(1, -1)


@lru_cache(maxsize=1)
def _load_visual_tcn(path: str | None) -> Optional[VisualTCN]:
    if not path:
        return None
    if not os.path.exists(path):
        warnings.warn(f"Visual TCN checkpoint not found at {path}")
        return None
    model = VisualTCN()
    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Failed to load Visual TCN from {path}: {exc}")
        return None


@lru_cache(maxsize=1)
def _load_wav2vec(model_name: str) -> Optional[AudioWav2VecEmbedder]:
    try:
        return AudioWav2VecEmbedder(model_name=model_name)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"wav2vec2 unavailable ({exc}); audio embeddings will be zeros")
        return None


def visual_tcn_embedding(lip_signal: np.ndarray, checkpoint: str | None) -> np.ndarray:
    model = _load_visual_tcn(checkpoint)
    if model is None or lip_signal.size == 0:
        return np.zeros((lip_signal.shape[0], 1), dtype=np.float32)
    with torch.no_grad():
        lip = torch.tensor(lip_signal, dtype=torch.float32).unsqueeze(0)  # (1, T)
        emb = model(lip).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        return emb.reshape(1, -1)


def wav2vec_embedding(
    waveform: np.ndarray,
    sr: int,
    model_name: str,
    checkpoint_path: str | None = None,
) -> np.ndarray:
    # checkpoint_path reserved for future fine-tuned wav2vec support
    _ = checkpoint_path
    embedder = _load_wav2vec(model_name)
    if embedder is None:
        return np.zeros((1, 1), dtype=np.float32)
    return embedder.embed(waveform, sr)


def efficientnet_b4_embedding(frames: list[np.ndarray], checkpoint_path: str | None = None) -> np.ndarray:
    """
    Placeholder EfficientNet-B4 embedding helper.
    Returns zeros when no model is available.
    """
    _ = checkpoint_path
    if not frames:
        return np.zeros((1, 1), dtype=np.float32)
    return np.zeros((len(frames), 1), dtype=np.float32)

