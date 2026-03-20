import os
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
except Exception:  # noqa: BLE001
    torchaudio = None  # Allow graceful degradation when torchaudio is missing

try:
    import torchvision
except Exception:  # noqa: BLE001
    torchvision = None  # Optional dependency for EfficientNet-B4 embeddings.

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

    def __init__(self, model_name: str = "WAV2VEC2_BASE", checkpoint_path: str | None = None):
        if torchaudio is None:
            raise RuntimeError("torchaudio is not available for wav2vec2 embeddings")
        try:
            bundle = getattr(torchaudio.pipelines, model_name)
        except AttributeError as exc:
            raise RuntimeError(f"torchaudio pipeline {model_name} not found") from exc

        self.model = bundle.get_model()
        if checkpoint_path:
            ckpt = os.path.expanduser(checkpoint_path)
            if os.path.exists(ckpt):
                try:
                    state = torch.load(ckpt, map_location="cpu")
                    if isinstance(state, dict):
                        model_state = state.get("model_state_dict", state)
                        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
                        if missing:
                            warnings.warn(
                                f"Wav2Vec2 checkpoint loaded with missing keys ({len(missing)}): {missing[:5]}"
                            )
                        if unexpected:
                            warnings.warn(
                                f"Wav2Vec2 checkpoint has unexpected keys ({len(unexpected)}): {unexpected[:5]}"
                            )
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"Failed to load wav2vec2 checkpoint at {ckpt}: {exc}")
            else:
                warnings.warn(f"Wav2Vec2 checkpoint path not found: {ckpt}")
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


class EfficientNetB4Embedder:
    """
    Frozen EfficientNet-B4 (ImageNet pretrained) frame embedder.

    Input frames are expected as uint8 RGB/BGR images; output is mean pooled
    feature embedding over the provided frame set.
    """

    def __init__(self):
        if torchvision is None:
            raise RuntimeError("torchvision is not available for EfficientNet-B4 embeddings")
        try:
            weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
            model = torchvision.models.efficientnet_b4(weights=weights)
            self._transform = weights.transforms()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load torchvision EfficientNet-B4 pretrained weights: {exc}") from exc

        self.model = model.features
        self.model.eval()

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((1, 1), dtype=np.float32)

        xs = []
        for frame in frames:
            if frame is None:
                continue
            arr = np.asarray(frame)
            if arr.size == 0:
                continue
            # Accept gray/BGR/RGB and coerce to uint8 RGB for torchvision transforms.
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 3:
                # OpenCV path usually provides BGR; convert to RGB.
                arr = arr[..., ::-1]
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
            pil = torchvision.transforms.functional.to_pil_image(arr)
            xs.append(self._transform(pil))

        if not xs:
            return np.zeros((1, 1), dtype=np.float32)

        x = torch.stack(xs, dim=0)
        with torch.no_grad():
            feats = self.model(x)
            pooled = F.adaptive_avg_pool2d(feats, output_size=(1, 1)).flatten(start_dim=1)
            emb = pooled.mean(dim=0, keepdim=True).cpu().numpy().astype(np.float32, copy=False)
        return emb


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


@lru_cache(maxsize=4)
def _load_wav2vec(model_name: str, checkpoint_path: str | None = None) -> Optional[AudioWav2VecEmbedder]:
    try:
        return AudioWav2VecEmbedder(model_name=model_name, checkpoint_path=checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"wav2vec2 unavailable ({exc}); audio embeddings will be zeros")
        return None


@lru_cache(maxsize=1)
def _load_effnet_b4() -> Optional[EfficientNetB4Embedder]:
    try:
        return EfficientNetB4Embedder()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"EfficientNet-B4 unavailable ({exc}); visual embeddings will be zeros")
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
    embedder = _load_wav2vec(model_name, checkpoint_path=checkpoint_path)
    if embedder is None:
        return np.zeros((1, 1), dtype=np.float32)
    return embedder.embed(waveform, sr)


def efficientnet_b4_embedding(frames: list[np.ndarray]) -> np.ndarray:
    embedder = _load_effnet_b4()
    if embedder is None:
        return np.zeros((1, 1), dtype=np.float32)
    return embedder.embed(frames)
