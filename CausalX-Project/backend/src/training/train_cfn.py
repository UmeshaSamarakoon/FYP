import argparse
import os
import random
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Sampler, TensorDataset, WeightedRandomSampler

from src.modules.causal_fusion import CausalFusionNetwork, CausalFusionNetworkV2
from src.cvi.feature_schema import (
    LIP_STREAM_FEATURES,
    NEXTGEN_AV_FEATURES,
    NEXTGEN_PHYS_FEATURES,
    resolve_feature_columns,
)

DEFAULT_SEED = 42
GEND_VISUAL_FEATURES = (
    "tcn_visual_emb",
    "effnet_b4_face_emb",
    "lip_roi_emb",
)
NEXTGEN_CRITICAL_FEATURES = tuple(
    dict.fromkeys(
        [
            "tcn_visual_emb",
            "wav2vec_audio_emb",
            "effnet_b4_face_emb",
            "lip_roi_emb",
            "wav2vec2_base_ft_emb",
            "video_motion_noise_ratio",
            "video_shape_noise_ratio",
            "video_temporal_instability",
            "video_detection_dropout",
            "video_compression_proxy",
            *list(NEXTGEN_AV_FEATURES),
            *list(NEXTGEN_PHYS_FEATURES),
        ]
    )
)


def resolve_lip_feature_columns(
    available_columns,
    enable_lip_stream: bool,
) -> list[str]:
    if not bool(enable_lip_stream):
        return []
    # Keep deterministic order; zero-fill missing columns in build_feature_matrix.
    _ = set(available_columns)
    return list(LIP_STREAM_FEATURES)


def _clip01_arr(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _robust_01(arr: np.ndarray, lo_q: float = 0.05, hi_q: float = 0.95) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.quantile(x, lo_q))
    hi = float(np.quantile(x, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return _clip01_arr((x - lo) / (hi - lo + 1e-8)).astype(np.float32, copy=False)


def infer_causal_breach_targets(
    df: pd.DataFrame,
    mode: str,
    column: str,
) -> tuple[np.ndarray | None, str]:
    """
    Build per-row causal-breach supervision targets in [0,1].

    mode:
      - none: disabled
      - column: use explicit dataset column if available, else fallback to heuristic
      - heuristic: derive target from AV mismatch + stability proxies
      - artifact_heuristic: derive target from AV mismatch + distortion/artifact proxies
    """
    resolved_mode = (mode or "none").strip().lower()
    if resolved_mode == "none":
        return None, "disabled"

    if resolved_mode == "column" and column in df.columns:
        vals = pd.to_numeric(df[column], errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)
        valid = (vals >= 0.0) & (vals <= 1.0)
        out = np.where(valid, vals, -1.0).astype(np.float32, copy=False)
        return out, f"column:{column}"

    # Heuristic fallback for datasets that do not carry causal_breach_score labels.
    corr = pd.to_numeric(df.get("av_correlation", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    peak = pd.to_numeric(df.get("av_peak_corr", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    lag = pd.to_numeric(df.get("av_lag_frames", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    jitter = pd.to_numeric(df.get("jitter_mean", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    mouth_flow = pd.to_numeric(df.get("mouth_flow_std", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    low_corr = 1.0 - _clip01_arr((corr + 1.0) / 2.0)
    low_peak = 1.0 - _clip01_arr((peak + 1.0) / 2.0)
    lag_mag = _robust_01(np.abs(lag))
    jitter_mag = _robust_01(np.abs(jitter))
    flow_mag = _robust_01(np.abs(mouth_flow))

    if resolved_mode == "artifact_heuristic":
        motion_noise = pd.to_numeric(df.get("video_motion_noise_ratio", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        shape_noise = pd.to_numeric(df.get("video_shape_noise_ratio", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        temporal_inst = pd.to_numeric(df.get("video_temporal_instability", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        dropout = pd.to_numeric(df.get("video_detection_dropout", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        compress = pd.to_numeric(df.get("video_compression_proxy", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        artifact = _clip01_arr(
            0.25 * _robust_01(np.abs(motion_noise))
            + 0.20 * _robust_01(np.abs(shape_noise))
            + 0.20 * _robust_01(np.abs(temporal_inst))
            + 0.20 * _robust_01(np.abs(dropout))
            + 0.15 * _robust_01(np.abs(compress))
        ).astype(np.float32, copy=False)
        heuristic = _clip01_arr(
            0.25 * low_corr
            + 0.15 * low_peak
            + 0.15 * lag_mag
            + 0.10 * jitter_mag
            + 0.10 * flow_mag
            + 0.25 * artifact
        ).astype(np.float32, copy=False)
    else:
        heuristic = _clip01_arr(
            0.30 * low_corr
            + 0.20 * low_peak
            + 0.20 * lag_mag
            + 0.15 * jitter_mag
            + 0.15 * flow_mag
        ).astype(np.float32, copy=False)

    if "label" in df.columns:
        label = pd.to_numeric(df["label"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        label = _clip01_arr(label)
        # Blend with label anchor so fake rows maintain higher breach priors.
        heuristic = _clip01_arr(0.55 * label + 0.45 * heuristic).astype(np.float32, copy=False)
    return heuristic, ("artifact_heuristic" if resolved_mode == "artifact_heuristic" else "heuristic")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairedRealFakeBatchSampler(Sampler[list[int]]):
    """
    Batch sampler that enforces paired real/fake sampling inside each batch.

    Each batch draws roughly half real (label=0) and half fake (label=1) indices.
    Optional sample_weights are used as per-class sampling probabilities.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        seed: int = DEFAULT_SEED,
        sample_weights: np.ndarray | None = None,
        pair_groups: np.ndarray | None = None,
    ) -> None:
        labels = np.asarray(labels).astype(int).reshape(-1)
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError("PairedRealFakeBatchSampler requires batch_size >= 2.")

        self.real_idx = np.where(labels == 0)[0].astype(np.int64)
        self.fake_idx = np.where(labels == 1)[0].astype(np.int64)
        if self.real_idx.size == 0 or self.fake_idx.size == 0:
            raise ValueError("PairedRealFakeBatchSampler requires both real and fake samples.")

        self.seed = int(seed)
        self._epoch = 0
        self.num_batches = int(np.ceil(len(labels) / float(self.batch_size)))
        self.pairs_per_batch = max(1, self.batch_size // 2)
        self.extra_per_batch = int(self.batch_size - 2 * self.pairs_per_batch)

        self.real_probs = None
        self.fake_probs = None
        self.paired_group_keys: np.ndarray | None = None
        self.paired_group_probs: np.ndarray | None = None
        self.group_real_idx: dict[object, np.ndarray] = {}
        self.group_fake_idx: dict[object, np.ndarray] = {}
        self.group_real_probs: dict[object, np.ndarray | None] = {}
        self.group_fake_probs: dict[object, np.ndarray | None] = {}
        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
            if sw.shape[0] == labels.shape[0]:
                r = np.clip(sw[self.real_idx], 1e-12, None)
                f = np.clip(sw[self.fake_idx], 1e-12, None)
                r_sum = float(np.sum(r))
                f_sum = float(np.sum(f))
                if np.isfinite(r_sum) and r_sum > 0.0:
                    self.real_probs = r / r_sum
                if np.isfinite(f_sum) and f_sum > 0.0:
                    self.fake_probs = f / f_sum

        # Optional same-source pairing: pair real/fake samples within the same group key.
        if pair_groups is not None:
            grp = np.asarray(pair_groups, dtype=object).reshape(-1)
            if grp.shape[0] == labels.shape[0]:
                by_group_real: dict[object, list[int]] = {}
                by_group_fake: dict[object, list[int]] = {}
                for idx, g in enumerate(grp.tolist()):
                    key = "__missing_group__" if g is None else str(g).strip()
                    key = key if key else "__missing_group__"
                    if int(labels[idx]) == 0:
                        by_group_real.setdefault(key, []).append(int(idx))
                    else:
                        by_group_fake.setdefault(key, []).append(int(idx))
                valid_keys: list[object] = []
                valid_mass: list[float] = []
                for key in sorted(set(list(by_group_real.keys()) + list(by_group_fake.keys()))):
                    real_rows = np.asarray(by_group_real.get(key, []), dtype=np.int64)
                    fake_rows = np.asarray(by_group_fake.get(key, []), dtype=np.int64)
                    if real_rows.size == 0 or fake_rows.size == 0:
                        continue
                    self.group_real_idx[key] = real_rows
                    self.group_fake_idx[key] = fake_rows
                    valid_keys.append(key)
                    valid_mass.append(float(min(real_rows.size, fake_rows.size)))

                    if sample_weights is not None and sw.shape[0] == labels.shape[0]:
                        wr = np.clip(sw[real_rows], 1e-12, None)
                        wf = np.clip(sw[fake_rows], 1e-12, None)
                        wr_sum = float(np.sum(wr))
                        wf_sum = float(np.sum(wf))
                        self.group_real_probs[key] = (wr / wr_sum) if np.isfinite(wr_sum) and wr_sum > 0.0 else None
                        self.group_fake_probs[key] = (wf / wf_sum) if np.isfinite(wf_sum) and wf_sum > 0.0 else None
                    else:
                        self.group_real_probs[key] = None
                        self.group_fake_probs[key] = None

                if valid_keys:
                    self.paired_group_keys = np.asarray(valid_keys, dtype=object)
                    mass = np.asarray(valid_mass, dtype=np.float64)
                    msum = float(np.sum(mass))
                    self.paired_group_probs = (mass / msum) if np.isfinite(msum) and msum > 0.0 else None

    def __len__(self) -> int:
        return int(self.num_batches)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        for _ in range(self.num_batches):
            if self.paired_group_keys is not None and self.paired_group_keys.size > 0:
                key_sel = rng.choice(
                    self.paired_group_keys,
                    size=self.pairs_per_batch,
                    replace=True,
                    p=self.paired_group_probs,
                )
                real_sel = np.empty(self.pairs_per_batch, dtype=np.int64)
                fake_sel = np.empty(self.pairs_per_batch, dtype=np.int64)
                for j, key in enumerate(key_sel.tolist()):
                    real_pool = self.group_real_idx[key]
                    fake_pool = self.group_fake_idx[key]
                    real_prob = self.group_real_probs.get(key)
                    fake_prob = self.group_fake_probs.get(key)
                    real_sel[j] = int(rng.choice(real_pool, size=1, replace=True, p=real_prob)[0])
                    fake_sel[j] = int(rng.choice(fake_pool, size=1, replace=True, p=fake_prob)[0])
            else:
                real_sel = rng.choice(
                    self.real_idx,
                    size=self.pairs_per_batch,
                    replace=True,
                    p=self.real_probs,
                )
                fake_sel = rng.choice(
                    self.fake_idx,
                    size=self.pairs_per_batch,
                    replace=True,
                    p=self.fake_probs,
                )

            batch = np.empty(2 * self.pairs_per_batch, dtype=np.int64)
            batch[0::2] = real_sel
            batch[1::2] = fake_sel

            if self.extra_per_batch > 0:
                # Keep the extra draw class-balanced in expectation.
                draw_real = bool(rng.random() < 0.5)
                pool = self.real_idx if draw_real else self.fake_idx
                probs = self.real_probs if draw_real else self.fake_probs
                extra = rng.choice(
                    pool,
                    size=self.extra_per_batch,
                    replace=True,
                    p=probs,
                )
                batch = np.concatenate([batch, extra.astype(np.int64)], axis=0)

            rng.shuffle(batch)
            yield batch.tolist()


def _resolve_gend_visual_indices(av_feature_cols: list[str]) -> list[int]:
    if not av_feature_cols:
        return []
    wanted = set(GEND_VISUAL_FEATURES)
    return [i for i, c in enumerate(av_feature_cols) if c in wanted]


def _apply_gend_visual_adaptation(
    av_features: torch.Tensor,
    labels: torch.Tensor,
    visual_indices: list[int],
    strength: float,
) -> torch.Tensor:
    """
    GenD-style visual adaptation:
    pair fake rows with real rows in-batch and move fake visual embeddings toward
    their paired real counterparts.
    """
    if av_features is None or av_features.ndim != 2 or not visual_indices:
        return av_features
    if float(strength) <= 0.0:
        return av_features

    y = labels.reshape(-1)
    fake_idx = torch.nonzero(y >= 0.5, as_tuple=False).squeeze(1)
    real_idx = torch.nonzero(y < 0.5, as_tuple=False).squeeze(1)
    n_pairs = int(min(fake_idx.numel(), real_idx.numel()))
    if n_pairs <= 0:
        return av_features

    fake_sel = fake_idx[torch.randperm(fake_idx.numel(), device=av_features.device)[:n_pairs]]
    real_sel = real_idx[torch.randperm(real_idx.numel(), device=av_features.device)[:n_pairs]]
    cols = torch.as_tensor(visual_indices, dtype=torch.long, device=av_features.device)

    out = av_features.clone()
    fake_feats = out[fake_sel][:, cols]
    real_feats = out[real_sel][:, cols]

    lam = torch.empty((n_pairs, 1), device=out.device, dtype=out.dtype).uniform_(
        0.0,
        float(np.clip(strength, 0.0, 1.0)),
    )
    adapted_fake = (1.0 - lam) * fake_feats + lam * real_feats
    out[fake_sel[:, None], cols[None, :]] = adapted_fake
    return out


def build_loaders(
    X_av,
    X_phys,
    y,
    batch_size,
    shuffle=True,
    X_lip=None,
    y_video=None,
    y_audio=None,
    y_causal_breach=None,
    weights=None,
    sampler=None,
    batch_sampler=None,
):
    tensors = [X_av, X_phys]
    if X_lip is not None:
        tensors.append(X_lip)
    tensors.append(y)
    if y_video is not None and y_audio is not None:
        tensors.extend([y_video, y_audio])
    if y_causal_breach is not None:
        tensors.append(y_causal_breach)
    if weights is not None:
        tensors.append(weights)
    dataset = TensorDataset(*tensors)
    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(bool(shuffle) and sampler is None),
            sampler=sampler,
        )
    return loader


def focal_loss_probs(preds, labels, alpha=0.75, gamma=2.0, eps=1e-7):
    preds = torch.clamp(preds, eps, 1.0 - eps)
    pt = torch.where(labels == 1, preds, 1.0 - preds)
    alpha_t = torch.where(labels == 1, torch.full_like(labels, alpha), torch.full_like(labels, 1.0 - alpha))
    return -alpha_t * ((1.0 - pt) ** gamma) * torch.log(pt)


def _compute_cls_loss(
    preds,
    label,
    criterion,
    loss_type="bce",
    focal_alpha=0.75,
    focal_gamma=2.0,
    focal_bce_mix: float | None = None,
):
    bce = criterion(preds, label)
    focal = focal_loss_probs(preds, label, alpha=focal_alpha, gamma=focal_gamma)
    if focal_bce_mix is None:
        return focal if loss_type == "focal" else bce
    mix = float(np.clip(focal_bce_mix, 0.0, 1.0))
    return (1.0 - mix) * bce + mix * focal


def _pairwise_ranking_loss(preds, labels, margin=0.2, max_pairs=1024):
    """
    Margin ranking surrogate for AUC:
      max(0, margin - (p_pos - p_neg))
    """
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    pos = preds[labels >= 0.5]
    neg = preds[labels < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return preds.new_tensor(0.0)

    max_pairs = int(max(1, max_pairs))
    # Subsample to keep memory bounded.
    max_pos = int(max(1, np.sqrt(max_pairs)))
    p_cnt = min(pos.numel(), max_pos)
    n_cnt = min(neg.numel(), max(1, max_pairs // max(1, p_cnt)))
    if p_cnt <= 0 or n_cnt <= 0:
        return preds.new_tensor(0.0)

    if pos.numel() > p_cnt:
        pos = pos[torch.randperm(pos.numel(), device=pos.device)[:p_cnt]]
    if neg.numel() > n_cnt:
        neg = neg[torch.randperm(neg.numel(), device=neg.device)[:n_cnt]]

    diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
    return torch.relu(float(margin) - diffs).mean()


def _effective_num_class_weights(labels: np.ndarray, beta: float = 0.9999) -> tuple[float, float]:
    """
    Class-balanced weighting using effective number of samples.
    Returns (w_neg, w_pos), normalized to mean=1.
    """
    y = np.asarray(labels).astype(int).reshape(-1)
    n_pos = max(int(np.sum(y == 1)), 1)
    n_neg = max(int(np.sum(y == 0)), 1)
    b = float(np.clip(beta, 0.0, 0.999999))

    def _w(n: int) -> float:
        if b <= 0.0:
            return 1.0 / float(max(n, 1))
        denom = 1.0 - (b ** float(max(n, 1)))
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            return 1.0
        return float((1.0 - b) / denom)

    w_neg = _w(n_neg)
    w_pos = _w(n_pos)
    m = 0.5 * (w_neg + w_pos)
    if not np.isfinite(m) or m <= 0.0:
        return 1.0, 1.0
    return float(w_neg / m), float(w_pos / m)


def _apply_class_weights_to_loss(
    raw_loss: torch.Tensor,
    labels: torch.Tensor,
    class_weights: tuple[float, float] | None,
) -> torch.Tensor:
    if class_weights is None:
        return raw_loss
    w_neg, w_pos = class_weights
    cls_w = torch.where(
        labels >= 0.5,
        torch.full_like(labels, float(w_pos)),
        torch.full_like(labels, float(w_neg)),
    )
    return raw_loss * cls_w


def _hyperspherical_margin_loss(
    embeddings: torch.Tensor | None,
    labels: torch.Tensor,
    margin: float = 0.35,
) -> torch.Tensor:
    """
    Encourage class clusters on the hypersphere while separating class centers.
    """
    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] < 2:
        base = labels.new_tensor(0.0, dtype=torch.float32)
        return base.to(embeddings.device if embeddings is not None else labels.device)

    z = F.normalize(embeddings, dim=1, eps=1e-8)
    y = labels.reshape(-1) >= 0.5
    pos = z[y]
    neg = z[~y]
    loss = z.new_tensor(0.0)

    center_pos = None
    center_neg = None
    if pos.numel() > 0:
        center_pos = F.normalize(pos.mean(dim=0, keepdim=True), dim=1, eps=1e-8)
        loss = loss + (1.0 - torch.matmul(pos, center_pos.t()).squeeze(1)).mean()
    if neg.numel() > 0:
        center_neg = F.normalize(neg.mean(dim=0, keepdim=True), dim=1, eps=1e-8)
        loss = loss + (1.0 - torch.matmul(neg, center_neg.t()).squeeze(1)).mean()
    if center_pos is not None and center_neg is not None:
        cos_centers = torch.sum(center_pos * center_neg, dim=1)
        separation_target = 1.0 - float(np.clip(margin, 0.0, 1.0))
        loss = loss + torch.relu(cos_centers - separation_target).mean()
    return loss


def _contrastive_pair_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    if z_a is None or z_b is None or z_a.ndim != 2 or z_b.ndim != 2:
        dev = z_a.device if z_a is not None else (z_b.device if z_b is not None else torch.device("cpu"))
        return torch.tensor(0.0, device=dev)
    n = int(min(z_a.shape[0], z_b.shape[0]))
    if n <= 1:
        return z_a.new_tensor(0.0)
    za = F.normalize(z_a[:n], dim=1, eps=1e-8)
    zb = F.normalize(z_b[:n], dim=1, eps=1e-8)
    logits = torch.matmul(za, zb.t()) / max(float(temperature), 1e-4)
    targets = torch.arange(n, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def _stage0_av_corr_epoch(
    model,
    loader,
    device,
    optimizer=None,
    temperature: float = 0.07,
    lip_weight: float = 0.25,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total = 0.0
    for batch in loader:
        av = batch[0].to(device)
        phys = batch[1].to(device)
        lip = batch[2].to(device) if len(batch) > 2 else None

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            if hasattr(model, "multitask_outputs"):
                out = model.multitask_outputs(av, phys, lip)
                av_h = out[1]
                phys_h = out[2]
                lip_h = out[3]
            else:
                _, av_h, phys_h, lip_h, _ = model.branch_outputs(av, phys, lip)

            loss = _contrastive_pair_loss(av_h, phys_h, temperature=temperature)
            if lip is not None and lip_h is not None and float(lip_weight) > 0.0:
                loss = loss + float(lip_weight) * 0.5 * (
                    _contrastive_pair_loss(av_h, lip_h, temperature=temperature)
                    + _contrastive_pair_loss(phys_h, lip_h, temperature=temperature)
                )

            if is_train:
                loss.backward()
                optimizer.step()
        total += float(loss.detach().item())
    return total / max(len(loader), 1)


def apply_temperature_scaling(probs, temperature: float) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return p.astype(np.float32, copy=False)
    t = float(max(temperature, 1e-4))
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p)) / t
    out = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(out, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)


def fit_temperature_scaler(
    labels,
    probs,
    t_min: float = 0.5,
    t_max: float = 3.0,
    num_steps: int = 61,
) -> dict[str, float]:
    y = np.asarray(labels).astype(np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if y.size == 0 or p.size == 0 or y.shape[0] != p.shape[0]:
        return {
            "temperature": 1.0,
            "nll_before": float("nan"),
            "nll_after": float("nan"),
        }

    p = np.clip(p, 1e-6, 1.0 - 1e-6)

    def _nll(prob_arr: np.ndarray) -> float:
        return float(-np.mean(y * np.log(prob_arr) + (1.0 - y) * np.log(1.0 - prob_arr)))

    nll_before = _nll(p)
    grid = np.linspace(float(t_min), float(t_max), int(max(3, num_steps)), dtype=np.float64)
    best_t = 1.0
    best_nll = nll_before
    for t in grid:
        pp = apply_temperature_scaling(p, temperature=float(t)).astype(np.float64)
        nll = _nll(pp)
        if nll < best_nll:
            best_nll = float(nll)
            best_t = float(t)
    return {
        "temperature": float(best_t),
        "nll_before": float(nll_before),
        "nll_after": float(best_nll),
    }


def _configure_ln_only_finetune(
    model: torch.nn.Module,
    unfreeze_heads: bool = True,
) -> int:
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            for p in module.parameters():
                p.requires_grad = True

    if bool(unfreeze_heads):
        prefixes = (
            "classifier.",
            "video_classifier.",
            "audio_classifier.",
            "causal_breach_head.",
            "alpha",
            "beta",
            "gamma",
        )
        for name, p in model.named_parameters():
            if name.startswith(prefixes) or name in {"alpha", "beta", "gamma"}:
                p.requires_grad = True

    return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))


def _masked_aux_loss(
    preds,
    labels,
    criterion,
    loss_type="bce",
    focal_alpha=0.75,
    focal_gamma=2.0,
    focal_bce_mix: float | None = None,
):
    if preds is None or labels is None:
        return None
    mask = labels >= 0
    if not bool(torch.any(mask)):
        return None
    p = preds[mask]
    y = labels[mask]
    aux_loss = _compute_cls_loss(
        p,
        y,
        criterion,
        loss_type=loss_type,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        focal_bce_mix=focal_bce_mix,
    )
    if aux_loss.dim() > 0:
        aux_loss = aux_loss.mean()
    return aux_loss


def _unpack_batch(
    batch,
    has_lip: bool = False,
    has_aux: bool = False,
    has_causal_breach_target: bool = False,
    has_weight: bool = False,
):
    """
    Supported batch layouts:
      [av, phys, y]
      [av, phys, lip, y]
      [+ y_video, y_audio]
      [+ causal_breach_target]
      [+ sample_weight]
    """
    idx = 0
    av = batch[idx]
    idx += 1
    phys = batch[idx]
    idx += 1

    lip = None
    if has_lip:
        lip = batch[idx]
        idx += 1

    label = batch[idx]
    idx += 1

    video_label = None
    audio_label = None
    causal_breach_target = None
    weight = None
    if has_aux:
        video_label = batch[idx]
        idx += 1
        audio_label = batch[idx]
        idx += 1
    if has_causal_breach_target:
        causal_breach_target = batch[idx]
        idx += 1
    if has_weight:
        weight = batch[idx]
        idx += 1
    return av, phys, lip, label, video_label, audio_label, causal_breach_target, weight


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_weights=False,
    use_causal=False,
    causal_weight=0.0,
    loss_type="bce",
    focal_alpha=0.75,
    focal_gamma=2.0,
    focal_bce_mix=None,
    enable_multitask=False,
    multitask_weight=0.0,
    ranking_loss_weight=0.0,
    ranking_margin=0.2,
    ranking_max_pairs=1024,
    class_balanced_weights=None,
    hyperspherical_reg_weight=0.0,
    hyperspherical_margin=0.35,
    has_lip=False,
    has_aux=False,
    has_causal_breach_target=False,
    has_weight=False,
    causal_breach_loss_weight=0.0,
    gend_visual_adapt=False,
    gend_adapt_strength=0.35,
    gend_adapt_prob=1.0,
    visual_adapt_indices=None,
):
    model.train()
    total_loss = 0.0
    for batch in loader:
        av, phys, lip, label, video_label, audio_label, causal_breach_target, weight = _unpack_batch(
            batch,
            has_lip=has_lip,
            has_aux=has_aux,
            has_causal_breach_target=has_causal_breach_target,
            has_weight=has_weight,
        )
        av = av.to(device)
        phys = phys.to(device)
        if lip is not None:
            lip = lip.to(device)
        label = label.to(device)
        if video_label is not None:
            video_label = video_label.to(device)
        if audio_label is not None:
            audio_label = audio_label.to(device)
        if causal_breach_target is not None:
            causal_breach_target = causal_breach_target.to(device)
        if weight is not None:
            weight = weight.to(device)

        if (
            bool(gend_visual_adapt)
            and visual_adapt_indices
            and float(gend_adapt_strength) > 0.0
            and (
                float(gend_adapt_prob) >= 1.0
                or float(torch.rand(1, device=av.device).item()) < float(gend_adapt_prob)
            )
        ):
            av = _apply_gend_visual_adaptation(
                av,
                label,
                visual_indices=list(visual_adapt_indices),
                strength=float(gend_adapt_strength),
            )

        optimizer.zero_grad()

        # Always run through branch outputs when we need latent or auxiliary heads.
        needs_branch = bool(
            use_causal
            or enable_multitask
            or ranking_loss_weight > 0
            or float(causal_breach_loss_weight) > 0.0
        )
        if needs_branch:
            if enable_multitask and hasattr(model, "multitask_outputs"):
                (
                    probs,
                    av_h,
                    phys_h,
                    lip_h,
                    video_probs,
                    audio_probs,
                    causal_breach_probs,
                ) = model.multitask_outputs(av, phys, lip)
            else:
                probs, av_h, phys_h, lip_h, causal_breach_probs = model.branch_outputs(av, phys, lip)
                video_probs = None
                audio_probs = None
        else:
            probs = model(av, phys, lip)
            av_h = None
            phys_h = None
            lip_h = None
            video_probs = None
            audio_probs = None
            causal_breach_probs = None

        raw_main = _compute_cls_loss(
            probs,
            label,
            criterion,
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            focal_bce_mix=focal_bce_mix,
        )
        raw_main = _apply_class_weights_to_loss(raw_main, label, class_balanced_weights)
        if use_weights and weight is not None:
            main_loss = (raw_main * weight).mean()
        else:
            main_loss = raw_main.mean() if raw_main.dim() > 0 else raw_main

        aux_loss = probs.new_tensor(0.0)
        if enable_multitask:
            aux_terms = []
            v_loss = _masked_aux_loss(
                video_probs,
                video_label,
                criterion,
                loss_type=loss_type,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                focal_bce_mix=focal_bce_mix,
            )
            if v_loss is not None:
                aux_terms.append(v_loss)
            a_loss = _masked_aux_loss(
                audio_probs,
                audio_label,
                criterion,
                loss_type=loss_type,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                focal_bce_mix=focal_bce_mix,
            )
            if a_loss is not None:
                aux_terms.append(a_loss)
            if aux_terms:
                aux_loss = torch.stack(aux_terms).mean()

        rank_loss = probs.new_tensor(0.0)
        if float(ranking_loss_weight) > 0.0:
            rank_loss = _pairwise_ranking_loss(
                probs,
                label,
                margin=ranking_margin,
                max_pairs=ranking_max_pairs,
            )

        causal_loss = probs.new_tensor(0.0)
        if use_causal and av_h is not None and phys_h is not None:
            causal_loss = model.causal_penalty(av_h, phys_h, lip_h)

        causal_breach_loss = probs.new_tensor(0.0)
        if (
            float(causal_breach_loss_weight) > 0.0
            and causal_breach_probs is not None
            and causal_breach_target is not None
        ):
            valid = causal_breach_target >= 0.0
            if bool(torch.any(valid)):
                cb_pred = causal_breach_probs[valid]
                cb_tgt = causal_breach_target[valid]
                cb_err = F.smooth_l1_loss(cb_pred, cb_tgt, reduction="none")
                if use_weights and weight is not None:
                    cb_w = weight[valid]
                    cb_err = cb_err * cb_w
                causal_breach_loss = cb_err.mean()

        hyper_loss = probs.new_tensor(0.0)
        if float(hyperspherical_reg_weight) > 0.0 and av_h is not None and phys_h is not None:
            fused_h = 0.5 * (av_h + phys_h)
            hyper_loss = _hyperspherical_margin_loss(
                fused_h,
                label,
                margin=hyperspherical_margin,
            )

        loss = (
            main_loss
            + float(causal_weight) * causal_loss
            + float(multitask_weight) * aux_loss
            + float(ranking_loss_weight) * rank_loss
            + float(causal_breach_loss_weight) * causal_breach_loss
            + float(hyperspherical_reg_weight) * hyper_loss
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def eval_epoch(
    model,
    loader,
    criterion,
    device,
    use_causal=False,
    causal_weight=0.0,
    loss_type="bce",
    focal_alpha=0.75,
    focal_gamma=2.0,
    focal_bce_mix=None,
    enable_multitask=False,
    multitask_weight=0.0,
    ranking_loss_weight=0.0,
    ranking_margin=0.2,
    ranking_max_pairs=1024,
    class_balanced_weights=None,
    hyperspherical_reg_weight=0.0,
    hyperspherical_margin=0.35,
    has_lip=False,
    has_aux=False,
    has_causal_breach_target=False,
    has_weight=False,
    causal_breach_loss_weight=0.0,
):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            av, phys, lip, label, video_label, audio_label, causal_breach_target, _ = _unpack_batch(
                batch,
                has_lip=has_lip,
                has_aux=has_aux,
                has_causal_breach_target=has_causal_breach_target,
                has_weight=has_weight,
            )
            av = av.to(device)
            phys = phys.to(device)
            if lip is not None:
                lip = lip.to(device)
            label = label.to(device)
            if video_label is not None:
                video_label = video_label.to(device)
            if audio_label is not None:
                audio_label = audio_label.to(device)
            if causal_breach_target is not None:
                causal_breach_target = causal_breach_target.to(device)

            needs_branch = bool(
                use_causal
                or enable_multitask
                or ranking_loss_weight > 0
                or float(causal_breach_loss_weight) > 0.0
            )
            if needs_branch:
                if enable_multitask and hasattr(model, "multitask_outputs"):
                    (
                        preds,
                        av_h,
                        phys_h,
                        lip_h,
                        video_preds,
                        audio_preds,
                        causal_breach_preds,
                    ) = model.multitask_outputs(av, phys, lip)
                else:
                    preds, av_h, phys_h, lip_h, causal_breach_preds = model.branch_outputs(av, phys, lip)
                    video_preds = None
                    audio_preds = None
            else:
                preds = model(av, phys, lip)
                av_h = None
                phys_h = None
                lip_h = None
                video_preds = None
                audio_preds = None
                causal_breach_preds = None

            raw_main = _compute_cls_loss(
                preds,
                label,
                criterion,
                loss_type=loss_type,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                focal_bce_mix=focal_bce_mix,
            )
            raw_main = _apply_class_weights_to_loss(raw_main, label, class_balanced_weights)
            main_loss = raw_main.mean() if raw_main.dim() > 0 else raw_main

            aux_loss = preds.new_tensor(0.0)
            if enable_multitask:
                aux_terms = []
                v_loss = _masked_aux_loss(
                    video_preds,
                    video_label,
                    criterion,
                    loss_type=loss_type,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    focal_bce_mix=focal_bce_mix,
                )
                if v_loss is not None:
                    aux_terms.append(v_loss)
                a_loss = _masked_aux_loss(
                    audio_preds,
                    audio_label,
                    criterion,
                    loss_type=loss_type,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    focal_bce_mix=focal_bce_mix,
                )
                if a_loss is not None:
                    aux_terms.append(a_loss)
                if aux_terms:
                    aux_loss = torch.stack(aux_terms).mean()

            rank_loss = preds.new_tensor(0.0)
            if float(ranking_loss_weight) > 0.0:
                rank_loss = _pairwise_ranking_loss(
                    preds,
                    label,
                    margin=ranking_margin,
                    max_pairs=ranking_max_pairs,
                )

            causal_loss = preds.new_tensor(0.0)
            if use_causal and av_h is not None and phys_h is not None:
                causal_loss = model.causal_penalty(av_h, phys_h, lip_h)

            causal_breach_loss = preds.new_tensor(0.0)
            if (
                float(causal_breach_loss_weight) > 0.0
                and causal_breach_preds is not None
                and causal_breach_target is not None
            ):
                valid = causal_breach_target >= 0.0
                if bool(torch.any(valid)):
                    causal_breach_loss = F.smooth_l1_loss(
                        causal_breach_preds[valid],
                        causal_breach_target[valid],
                        reduction="mean",
                    )

            hyper_loss = preds.new_tensor(0.0)
            if float(hyperspherical_reg_weight) > 0.0 and av_h is not None and phys_h is not None:
                fused_h = 0.5 * (av_h + phys_h)
                hyper_loss = _hyperspherical_margin_loss(
                    fused_h,
                    label,
                    margin=hyperspherical_margin,
                )

            loss = (
                main_loss
                + float(causal_weight) * causal_loss
                + float(multitask_weight) * aux_loss
                + float(ranking_loss_weight) * rank_loss
                + float(causal_breach_loss_weight) * causal_breach_loss
                + float(hyperspherical_reg_weight) * hyper_loss
            )
            total_loss += loss.item()
            all_preds.extend(preds.squeeze(1).cpu().numpy().tolist())
            all_labels.extend(label.squeeze(1).cpu().numpy().tolist())

    if not all_labels:
        return 0.0, 0.0, 0.0, [], []

    acc = accuracy_score(all_labels, [1 if p >= 0.5 else 0 for p in all_preds])
    try:
        auc_val = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc_val = 0.0
    return total_loss / max(len(loader), 1), acc, auc_val, all_labels, all_preds


def threshold_sweep(labels, probs):
    """
    Sweep thresholds to find best F1 and report PR AUC.
    """
    labels = np.array(labels)
    probs = np.array(probs)
    if probs.size == 0 or labels.size == 0:
        return {"pr_auc": 0.0, "best_f1": 0.0, "best_thr": 0.5}

    prec, rec, thr = precision_recall_curve(labels, probs)
    pr_auc = auc(rec, prec)

    best_f1, best_thr = 0.0, 0.5
    thresholds = np.linspace(probs.min(), probs.max(), 50) if probs.size else [0.5]
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    return {"pr_auc": pr_auc, "best_f1": best_f1, "best_thr": best_thr}


def _build_threshold_grid(probs, max_points=301):
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return np.array([0.5], dtype=float)

    lo = float(np.min(probs))
    hi = float(np.max(probs))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.array([0.5], dtype=float)
    if abs(hi - lo) < 1e-12:
        return np.array([float(np.clip(lo, 0.0, 1.0))], dtype=float)

    lin = np.linspace(lo, hi, max_points, dtype=float)
    qn = min(max_points, 101)
    quant = np.quantile(probs, np.linspace(0.0, 1.0, qn))
    grid = np.unique(np.concatenate([lin, quant, np.array([0.5], dtype=float)]))
    return np.clip(grid, 0.0, 1.0)


def calibrate_threshold_to_targets(labels, probs, targets, priority="f1"):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    active_targets = {k: float(v) for k, v in (targets or {}).items() if v is not None}

    priority_key = (priority or "f1").strip().lower()
    if priority_key == "accuracy":
        priority_key = "acc"
    elif priority_key == "precision":
        priority_key = "prec"
    elif priority_key == "recall":
        priority_key = "rec"
    elif priority_key in {"balanced", "balanced_acc", "bal_acc"}:
        priority_key = "bal_acc"
    if priority_key not in {"acc", "prec", "rec", "f1", "bal_acc"}:
        priority_key = "f1"

    if labels.size == 0 or probs.size == 0:
        return {
            "threshold": 0.5,
            "metrics": {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "bal_acc": 0.0, "spec": 0.0},
            "meets_all_targets": False,
            "target_shortfall": float(sum(active_targets.values())),
            "targets": active_targets,
            "priority_metric": priority_key,
            "num_feasible": 0,
        }

    grid = _build_threshold_grid(probs, max_points=301)
    best_meet = None
    best_fallback = None
    feasible_count = 0

    for t in grid:
        preds = (probs >= float(t)).astype(int)
        m = _binary_confusion(labels, preds)
        shortfall = float(sum(max(0.0, float(tv) - float(m.get(k, 0.0))) for k, tv in active_targets.items()))

        metric_sort = (
            float(m[priority_key]),
            float(m["f1"]),
            float(m["bal_acc"]),
            -abs(float(t) - 0.5),
        )

        if shortfall <= 1e-12:
            feasible_count += 1
            if best_meet is None or metric_sort > best_meet["sort_key"]:
                best_meet = {
                    "threshold": float(t),
                    "metrics": m,
                    "target_shortfall": 0.0,
                    "sort_key": metric_sort,
                }
        else:
            fallback_sort = (-shortfall,) + metric_sort
            if best_fallback is None or fallback_sort > best_fallback["sort_key"]:
                best_fallback = {
                    "threshold": float(t),
                    "metrics": m,
                    "target_shortfall": shortfall,
                    "sort_key": fallback_sort,
                }

    chosen = best_meet if best_meet is not None else best_fallback
    if chosen is None:
        chosen = {
            "threshold": 0.5,
            "metrics": _binary_confusion(labels, (probs >= 0.5).astype(int)),
            "target_shortfall": float(sum(active_targets.values())),
        }

    return {
        "threshold": float(chosen["threshold"]),
        "metrics": {k: float(v) for k, v in chosen["metrics"].items()},
        "meets_all_targets": bool(best_meet is not None),
        "target_shortfall": float(chosen["target_shortfall"]),
        "targets": active_targets,
        "priority_metric": priority_key,
        "num_feasible": int(feasible_count),
    }


def infer_domain_labels(df: pd.DataFrame):
    if "dataset" in df.columns:
        return df["dataset"].astype(str).str.lower().fillna("unknown").to_numpy()
    if "path" in df.columns:
        path_s = df["path"].astype(str).str.lower()
        dom = np.full(len(df), "unknown", dtype=object)
        dom[path_s.str.contains("fakeavceleb", regex=False).to_numpy()] = "fakeavceleb"
        dom[path_s.str.contains("dfdc", regex=False).to_numpy()] = "dfdc"
        return dom
    return np.array(["unknown"] * len(df), dtype=object)


def build_feature_matrix(df: pd.DataFrame, columns: list[str], name: str):
    cols = []
    missing = []
    for c in columns:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
            cols.append(s.to_numpy())
        else:
            missing.append(c)
            cols.append(np.zeros(len(df), dtype=np.float32))
    if missing:
        print(f"{name}: zero-filled {len(missing)} missing columns: {missing}")
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.column_stack(cols).astype(np.float32)


def _expected_nextgen_columns(use_embeddings: bool, required_mode: str) -> list[str]:
    mode = (required_mode or "critical").strip().lower()
    if mode == "none":
        return []
    if mode == "all":
        av_cols, phys_cols = resolve_feature_columns(
            [],
            use_embeddings=bool(use_embeddings),
            profile="nextgen",
        )
        return list(dict.fromkeys(list(av_cols) + list(phys_cols)))
    return list(NEXTGEN_CRITICAL_FEATURES)


def _validate_feature_coverage(
    df: pd.DataFrame,
    feature_profile: str,
    use_embeddings: bool,
    required_mode: str,
    policy: str,
) -> None:
    if str(feature_profile).strip().lower() != "nextgen":
        return
    resolved_policy = (policy or "warn").strip().lower()
    if resolved_policy == "off":
        return
    required_cols = _expected_nextgen_columns(
        use_embeddings=bool(use_embeddings),
        required_mode=str(required_mode),
    )
    if not required_cols:
        return

    missing = [c for c in required_cols if c not in df.columns]
    empty = []
    for c in required_cols:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if int(vals.notna().sum()) <= 0:
            empty.append(c)

    issues = []
    if missing:
        issues.append(f"missing={missing}")
    if empty:
        issues.append(f"all_nan={empty}")
    if not issues:
        return

    msg = (
        "Insufficient nextgen feature coverage for --feature-profile=nextgen "
        f"({', '.join(issues)}). "
        "Use an enriched processed CSV (for example, "
        "data/processed/causal_multimodal_dataset_effnet_w2v2_physfix_fullvis_proxy_nextgen.csv) "
        "or relax checks with --missing-feature-policy warn/off."
    )
    if resolved_policy == "error":
        raise RuntimeError(msg)
    print("Warning:", msg)


def rebalance_by_label(df: pd.DataFrame, mode: str = "none", label_col: str = "label", seed: int = 42):
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return df
    if label_col not in df.columns:
        print(f"Class balancing skipped: missing '{label_col}' column.")
        return df

    counts = df[label_col].astype(int).value_counts()
    if len(counts) < 2:
        print("Class balancing skipped: single class only.")
        return df

    if mode == "downsample":
        target = int(counts.min())
        parts = [
            g.sample(n=target, random_state=seed, replace=False)
            for _, g in df.groupby(label_col)
        ]
    elif mode == "upsample":
        target = int(counts.max())
        parts = [
            g.sample(n=target, random_state=seed, replace=(len(g) < target))
            for _, g in df.groupby(label_col)
        ]
    else:
        raise ValueError(f"Unsupported --class-balance-mode value: {mode}")

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    after_counts = out[label_col].astype(int).value_counts().to_dict()
    print(f"Applied {mode} class balancing: {counts.to_dict()} -> {after_counts}")
    return out


def split_train_val_indices(
    df: pd.DataFrame,
    val_split: float,
    seed: int,
    domain_labels: np.ndarray,
    group_col: str | None = None,
):
    labels = df["label"].astype(int).to_numpy()
    all_idx = np.arange(len(df))

    group_col = (group_col or "").strip()
    if group_col and group_col in df.columns:
        raw_groups = df[group_col].astype(str)
        # Missing/blank group ids should not collapse unrelated rows together.
        groups = np.where(
            raw_groups.str.strip().to_numpy() != "",
            raw_groups.to_numpy(),
            np.array([f"__row_{i}" for i in range(len(df))], dtype=object),
        )

        grp_df = pd.DataFrame(
            {
                "group": groups,
                "label": labels,
                "domain": np.asarray(domain_labels, dtype=object),
            }
        )
        grp_agg = grp_df.groupby("group", as_index=False).agg(
            label=("label", "max"),
            domain=("domain", "first"),
        )

        group_combo = np.array(
            [f"{d}:{int(y)}" for d, y in zip(grp_agg["domain"].to_numpy(), grp_agg["label"].to_numpy())],
            dtype=object,
        )
        combo_counts = pd.Series(group_combo).value_counts()
        if len(combo_counts) > 1 and int(combo_counts.min()) >= 2:
            stratify_groups = group_combo
            print(f"Using group+domain+label stratified split by {group_col} across {len(combo_counts)} groups.")
        elif len(np.unique(grp_agg["label"].to_numpy())) > 1:
            stratify_groups = grp_agg["label"].to_numpy()
            print(f"Using group+label stratified split by {group_col}.")
        else:
            stratify_groups = None
            print(f"Group split by {group_col} has no stratification (single class).")

        train_groups, val_groups = train_test_split(
            grp_agg["group"].to_numpy(),
            test_size=val_split,
            random_state=seed,
            stratify=stratify_groups,
        )
        idx_train = np.where(np.isin(groups, train_groups))[0]
        idx_val = np.where(np.isin(groups, val_groups))[0]
        print(f"Group split on '{group_col}': train_groups={len(train_groups)} val_groups={len(val_groups)}")
        return idx_train, idx_val

    stratify_target = labels if len(np.unique(labels)) > 1 else None
    domain_label_combo = np.array([f"{d}:{int(y)}" for d, y in zip(domain_labels, labels)], dtype=object)
    combo_counts = pd.Series(domain_label_combo).value_counts()
    if len(combo_counts) > 1 and int(combo_counts.min()) >= 2:
        stratify_target = domain_label_combo
        print(f"Using dataset+label stratified split across {len(combo_counts)} groups.")
    elif stratify_target is not None:
        print("Using label-only stratified split.")
    else:
        print("Split has no stratification (single class).")

    idx_train, idx_val = train_test_split(
        all_idx,
        test_size=val_split,
        random_state=seed,
        stratify=stratify_target,
    )
    return idx_train, idx_val


def _binary_confusion(labels, preds):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    rec = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    bal = 0.5 * (rec + spec)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": acc,
        "rec": rec,
        "spec": spec,
        "prec": prec,
        "bal_acc": bal,
        "f1": f1,
    }


def compute_domain_metrics(labels, probs, domains, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    domains = np.asarray(domains, dtype=object)
    preds = (probs >= float(threshold)).astype(int)

    overall = _binary_confusion(labels, preds)
    per_domain = {}
    for d in sorted(set(domains.tolist())):
        mask = domains == d
        if not np.any(mask):
            continue
        per_domain[str(d)] = _binary_confusion(labels[mask], preds[mask])

    if per_domain:
        bal_list = [m["bal_acc"] for m in per_domain.values()]
        spec_list = [m["spec"] for m in per_domain.values()]
        rec_list = [m["rec"] for m in per_domain.values()]
        macro_bal = float(np.mean(bal_list))
        worst_bal = float(np.min(bal_list))
        worst_spec = float(np.min(spec_list))
        worst_rec = float(np.min(rec_list))
    else:
        macro_bal = worst_bal = overall["bal_acc"]
        worst_spec = overall["spec"]
        worst_rec = overall["rec"]

    return {
        "overall": overall,
        "per_domain": per_domain,
        "macro_bal_acc": macro_bal,
        "worst_bal_acc": worst_bal,
        "worst_spec": worst_spec,
        "worst_rec": worst_rec,
    }


def filter_training_sources(df, train_source):
    """
    Keep only requested data source(s) for training.
    """
    source = (train_source or "all").strip().lower()
    if source == "all":
        return df

    if source != "fakeavceleb":
        raise ValueError(f"Unsupported --train-source value: {train_source}")

    if "dataset" in df.columns:
        mask = df["dataset"].astype(str).str.lower().eq("fakeavceleb")
        return df.loc[mask].copy()

    if "path" in df.columns:
        mask = df["path"].astype(str).str.lower().str.contains("fakeavceleb", regex=False)
        return df.loc[mask].copy()

    if "video_id" in df.columns:
        # Conservative fallback: keep rows whose video_id explicitly mentions fakeavceleb.
        mask = df["video_id"].astype(str).str.lower().str.contains("fakeavceleb", regex=False)
        filtered = df.loc[mask].copy()
        if not filtered.empty:
            return filtered

    raise RuntimeError(
        "Could not isolate FakeAVCeleb rows: missing 'dataset' or 'path' hints in input CSV."
    )


def _norm_path(v):
    try:
        return Path(str(v)).expanduser().as_posix().lower()
    except Exception:
        return str(v).strip().lower()


def _norm_name(v):
    try:
        return Path(str(v)).name.lower()
    except Exception:
        return str(v).strip().lower()


def _resolve_error_lookup_paths(train_df: pd.DataFrame):
    """
    Returns (mode, keys) where:
      mode in {"path", "video_id", "none"}
      keys is a normalized string Series aligned to train_df rows.
    """
    if "path" in train_df.columns:
        return "path", train_df["path"].astype(str).map(_norm_path)
    if "video_id" in train_df.columns:
        return "video_id", train_df["video_id"].astype(str).map(_norm_name)
    return "none", pd.Series([""] * len(train_df), index=train_df.index)


def _normalize_lookup_paths(paths, mode: str):
    mode = (mode or "none").strip().lower()
    if mode == "path":
        return set(_norm_path(p) for p in paths)
    if mode == "video_id":
        return set(_norm_name(Path(str(p)).name) for p in paths)
    return set()


def load_error_paths(path: str, true_label: int | None = None, pred_label: int | None = None):
    """
    Load error paths from jsonl/csv/tsv and return a normalized path set.
    Expected rows typically contain at least: path, label, pred.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Error-path file not found: {path}")

    want_true = None if true_label is None else int(true_label)
    want_pred = None if pred_label is None else int(pred_label)

    def _row_match(label, pred):
        if label is None or pred is None:
            return True
        try:
            lbl_i = int(label)
            pred_i = int(pred)
        except Exception:
            return True
        if want_true is not None and lbl_i != want_true:
            return False
        if want_pred is not None and pred_i != want_pred:
            return False
        return True

    paths = set()
    raw_lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if p.suffix.lower() == ".jsonl":
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            row_path = row.get("path")
            if not row_path:
                continue
            label = row.get("label")
            pred = row.get("pred")
            if _row_match(label, pred):
                paths.add(_norm_path(row_path))
    else:
        # Common notebook output: plain text file with one absolute path per line.
        # Avoid delimiter sniffing because video paths can contain punctuation/spaces.
        if raw_lines and all(("\t" not in ln and "," not in ln) for ln in raw_lines[:32]):
            return set(_norm_path(ln) for ln in raw_lines)

        # First try regular CSV/TSV with headers.
        try:
            df = pd.read_csv(p, sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            df = None
        if df is not None and "path" in df.columns:
            if "label" in df.columns and want_true is not None:
                df = df[pd.to_numeric(df["label"], errors="coerce") == float(want_true)]
            if "pred" in df.columns and want_pred is not None:
                df = df[pd.to_numeric(df["pred"], errors="coerce") == float(want_pred)]
            paths = set(df["path"].astype(str).map(_norm_path).tolist())
            return paths

        # Fallback for headerless TSV/CSV produced by notebook scripts:
        # path<TAB>label<TAB>pred  (or path,label,pred)
        for line in raw_lines:
            if "\t" in line:
                parts = line.rsplit("\t", 2)
            else:
                parts = line.rsplit(",", 2)

            if not parts:
                continue
            row_path = parts[0].strip()
            if not row_path or row_path.lower() == "path":
                continue

            if len(parts) >= 3:
                try:
                    label = int(parts[1].strip())
                    pred = int(parts[2].strip())
                except Exception:
                    paths.add(_norm_path(row_path))
                    continue
                if _row_match(label, pred):
                    paths.add(_norm_path(row_path))
            else:
                paths.add(_norm_path(row_path))

    return paths


def load_hard_negative_paths(path: str):
    """
    Real (label=0) predicted fake (pred=1).
    """
    return load_error_paths(path, true_label=0, pred_label=1)


def load_hard_positive_paths(path: str):
    """
    Fake (label=1) predicted real (pred=0).
    """
    return load_error_paths(path, true_label=1, pred_label=0)


def _scenario_focus_mask(df: pd.DataFrame, scenario_focus: str):
    """
    Build a mask over rows for optional scenario-aware upweighting.
    Returns (mask, status) where status describes availability.
    """
    mode = (scenario_focus or "none").strip().lower()
    if mode == "none":
        return np.zeros(len(df), dtype=bool), "none"
    if "video_fake" not in df.columns or "audio_fake" not in df.columns:
        return np.zeros(len(df), dtype=bool), "missing_video_audio_cols"

    video = pd.to_numeric(df["video_fake"], errors="coerce")
    audio = pd.to_numeric(df["audio_fake"], errors="coerce")
    valid = video.isin([0.0, 1.0]) & audio.isin([0.0, 1.0])

    if mode == "video_only_fake":
        cond = (video == 1.0) & (audio == 0.0)
    elif mode == "audio_only_fake":
        cond = (video == 0.0) & (audio == 1.0)
    elif mode == "both_fake":
        cond = (video == 1.0) & (audio == 1.0)
    else:
        raise ValueError(
            "Unsupported --scenario-focus value: "
            f"{scenario_focus}. Expected one of: none, video_only_fake, audio_only_fake, both_fake."
        )
    return (valid & cond).to_numpy(), "ok"


def _resolve_pair_groups(df: pd.DataFrame, source_col: str) -> tuple[np.ndarray | None, str]:
    def _extract_pair_key(v) -> str:
        s = str(v or "").strip()
        if not s:
            return ""
        stem = Path(s).name.rsplit(".", 1)[0]
        tail = stem.split("_", 1)[1] if "_" in stem else stem
        m = re.search(r"(id\d+[_-]\d+)", tail, flags=re.IGNORECASE)
        if m is not None:
            return m.group(1).replace("-", "_").lower()
        parts = [p for p in re.split(r"[_\s]+", tail) if p]
        if len(parts) >= 2:
            return f"{parts[0].lower()}_{parts[1].lower()}"
        if parts:
            return parts[0].lower()
        return tail.lower()

    col = (source_col or "").strip()
    if col and col in df.columns:
        vals = df[col].astype(str).fillna("").to_numpy(dtype=object)
        vals = np.array([v if str(v).strip() else f"__row_{i}" for i, v in enumerate(vals)], dtype=object)
        return vals, f"column:{col}"
    if "pair_source_id" in df.columns:
        vals = df["pair_source_id"].astype(str).fillna("").to_numpy(dtype=object)
        vals = np.array([v if str(v).strip() else f"__row_{i}" for i, v in enumerate(vals)], dtype=object)
        return vals, "column:pair_source_id"
    if "video_id" in df.columns or "path" in df.columns:
        out = None
        if "video_id" in df.columns:
            out = df["video_id"].astype(str).map(_extract_pair_key)
        if "path" in df.columns:
            from_path = df["path"].astype(str).map(_extract_pair_key)
            if out is None:
                out = from_path
            else:
                out = out.where(out.astype(str).str.len() > 0, from_path)
        if out is not None:
            vals = out.fillna("").astype(str).to_numpy(dtype=object)
            vals = np.array([v if str(v).strip() else f"__row_{i}" for i, v in enumerate(vals)], dtype=object)
            return vals, "inferred:video_id/path"
    return None, "disabled"


def _resolve_generator_groups(df: pd.DataFrame, group_col: str) -> tuple[np.ndarray | None, str]:
    col = (group_col or "").strip()
    if col and col in df.columns:
        vals = df[col].astype(str).fillna("unknown").to_numpy(dtype=object)
        return vals, f"column:{col}"
    if {"dataset", "video_fake", "audio_fake"}.issubset(df.columns):
        dataset = df["dataset"].astype(str).fillna("unknown")
        vf = pd.to_numeric(df["video_fake"], errors="coerce").fillna(-1).astype(int).astype(str)
        af = pd.to_numeric(df["audio_fake"], errors="coerce").fillna(-1).astype(int).astype(str)
        vals = (dataset + ":" + vf + ":" + af).to_numpy(dtype=object)
        return vals, "dataset:video_fake:audio_fake"
    if "dataset" in df.columns:
        return df["dataset"].astype(str).fillna("unknown").to_numpy(dtype=object), "column:dataset"
    return None, "unavailable"


def _generator_balance_weights(
    groups: np.ndarray | None,
    mode: str,
    beta: float,
) -> np.ndarray | None:
    if groups is None:
        return None
    vals = np.asarray(groups, dtype=object).reshape(-1)
    if vals.size == 0:
        return None

    vc = pd.Series(vals).value_counts()
    if vc.empty:
        return None
    resolved_mode = (mode or "inverse_freq").strip().lower()
    w_map: dict[object, float] = {}
    for key, count in vc.items():
        n = max(int(count), 1)
        if resolved_mode == "effective_num":
            b = float(np.clip(beta, 0.0, 0.999999))
            denom = 1.0 - (b ** float(n))
            w = (1.0 - b) / denom if np.isfinite(denom) and abs(denom) > 1e-12 else 1.0
        else:
            w = 1.0 / float(n)
        w_map[key] = float(max(w, 1e-8))

    out = np.asarray([w_map.get(v, 1.0) for v in vals], dtype=np.float64)
    mu = float(np.mean(out))
    if np.isfinite(mu) and mu > 0.0:
        out = out / mu
    return out.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Train Causal Fusion Network (CFN).")
    parser.add_argument(
        "--data",
        "--dataset-csv",
        dest="data",
        default="data/processed/causal_multimodal_dataset.csv",
        help="Path to the dataset CSV (default: data/processed/causal_multimodal_dataset.csv)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="Optional checkpoint path used to initialize model weights before training.",
    )
    parser.add_argument(
        "--init-strict",
        action="store_true",
        help="Load --init-model with strict=True (default False allows head-shape differences).",
    )
    parser.add_argument("--use-scaler", action="store_true")
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Include TCN/Wav2Vec2 embedding columns and train CFN V2.",
    )
    parser.add_argument(
        "--enable-lip-stream",
        action="store_true",
        help=(
            "Enable a third lip/mouth visual stream branch (cross-modal attention over AV, physical, lip). "
            "Lip columns are zero-filled when absent."
        ),
    )
    parser.add_argument(
        "--causal-weight",
        type=float,
        default=0.0,
        help="Weight for SCM-inspired consistency penalty (0 to disable).",
    )
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "cosine"],
        default="cosine",
        help="Learning rate scheduler to use (default: cosine).",
    )
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--focal-bce-mix",
        type=float,
        default=None,
        help=(
            "Optional focal/BCE interpolation in [0,1]. "
            "0 => pure BCE, 1 => pure focal. Overrides --loss behavior when set."
        ),
    )
    parser.add_argument(
        "--class-balanced-loss",
        action="store_true",
        help="Enable class-balanced effective-number loss weighting on top of the main classification loss.",
    )
    parser.add_argument(
        "--class-balance-beta",
        type=float,
        default=0.9999,
        help="Beta parameter for class-balanced effective-number weighting (typically close to 1).",
    )
    parser.add_argument(
        "--hyperspherical-reg-weight",
        type=float,
        default=0.0,
        help="Weight for hyperspherical feature regularization loss.",
    )
    parser.add_argument(
        "--hyperspherical-margin",
        type=float,
        default=0.35,
        help="Class-center angular margin proxy used by hyperspherical regularization.",
    )
    parser.add_argument(
        "--enable-av-layernorm-adapter",
        action="store_true",
        help="Enable AV input LayerNorm adapter block (used for LN-only parameter-efficient adaptation).",
    )
    parser.add_argument(
        "--ln-only-finetune",
        action="store_true",
        help="Freeze non-LayerNorm parameters and fine-tune LayerNorm affine params (+heads by default).",
    )
    parser.add_argument(
        "--ln-only-unfreeze-heads",
        action="store_true",
        help="When --ln-only-finetune is enabled, also keep classifier/fusion scalars trainable.",
    )
    parser.add_argument(
        "--stage0-avc-pretrain",
        action="store_true",
        help="Run Stage-0 AV correspondence pretraining (contrastive AV alignment) before supervised fine-tuning.",
    )
    parser.add_argument(
        "--stage0-epochs",
        type=int,
        default=8,
        help="Epochs for Stage-0 AV correspondence pretraining.",
    )
    parser.add_argument(
        "--stage0-patience",
        type=int,
        default=3,
        help="Patience for Stage-0 AV correspondence pretraining.",
    )
    parser.add_argument(
        "--stage0-lr",
        type=float,
        default=1e-4,
        help="Learning rate for Stage-0 AV correspondence pretraining.",
    )
    parser.add_argument(
        "--stage0-temperature",
        type=float,
        default=0.07,
        help="InfoNCE temperature for Stage-0 AV correspondence pretraining.",
    )
    parser.add_argument(
        "--stage0-lip-weight",
        type=float,
        default=0.25,
        help="Weight for optional lip correspondence terms during Stage-0 pretraining.",
    )
    parser.add_argument(
        "--stage0-real-only",
        action="store_true",
        help="Use only real rows for Stage-0 AV correspondence pretraining.",
    )
    parser.add_argument(
        "--enable-multitask",
        action="store_true",
        help=(
            "Enable auxiliary heads for video_fake/audio_fake prediction "
            "when those columns are present."
        ),
    )
    parser.add_argument(
        "--multitask-weight",
        type=float,
        default=0.0,
        help="Weight for auxiliary multi-task BCE/Focal losses.",
    )
    parser.add_argument(
        "--ranking-loss-weight",
        type=float,
        default=0.0,
        help="Weight for pairwise ranking loss surrogate (AUC-oriented).",
    )
    parser.add_argument(
        "--ranking-margin",
        type=float,
        default=0.2,
        help="Margin used by pairwise ranking loss.",
    )
    parser.add_argument(
        "--ranking-max-pairs",
        type=int,
        default=1024,
        help="Upper bound for sampled pos-neg pairs per batch in ranking loss.",
    )
    parser.add_argument(
        "--joint-ce-auc-margin",
        action="store_true",
        help=(
            "Enable joint CE + AUC-margin training objective. "
            "If no explicit margin weight is set, defaults margin-loss weight to 0.2."
        ),
    )
    parser.add_argument(
        "--auc-margin-loss-weight",
        type=float,
        default=None,
        help="Alias for --ranking-loss-weight.",
    )
    parser.add_argument(
        "--auc-margin",
        type=float,
        default=None,
        help="Alias for --ranking-margin.",
    )
    parser.add_argument(
        "--auc-margin-max-pairs",
        type=int,
        default=None,
        help="Alias for --ranking-max-pairs.",
    )
    parser.add_argument(
        "--causal-breach-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for auxiliary causal-breach supervision head loss. "
            "Set >0 to optimize causal-breach predictions in addition to fake classification."
        ),
    )
    parser.add_argument(
        "--causal-breach-target",
        choices=["none", "column", "heuristic", "artifact_heuristic"],
        default="none",
        help=(
            "Source of causal-breach supervision targets. "
            "'column' uses --causal-breach-column when available, else falls back to heuristic."
        ),
    )
    parser.add_argument(
        "--causal-breach-column",
        type=str,
        default="causal_breach_score",
        help="Column used when --causal-breach-target=column.",
    )
    parser.add_argument(
        "--distortion-loss-weight",
        type=float,
        default=None,
        help="Alias for --causal-breach-loss-weight (D3-style discrepancy/distortion auxiliary branch).",
    )
    parser.add_argument(
        "--distortion-target",
        type=str,
        default=None,
        choices=["none", "column", "heuristic", "artifact_heuristic"],
        help="Alias for --causal-breach-target.",
    )
    parser.add_argument(
        "--distortion-column",
        type=str,
        default=None,
        help="Alias for --causal-breach-column.",
    )
    parser.add_argument(
        "--train-source",
        choices=["fakeavceleb", "all"],
        default="fakeavceleb",
        help="Training source filter. Default keeps only FakeAVCeleb rows.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=["baseline", "extended", "nextgen", "auto"],
        default="auto",
        help=(
            "Feature profile for AV/physical inputs. "
            "'baseline' keeps legacy dims, 'extended' forces richer feature set, "
            "'nextgen' enables cross-modal discrepancy/distortion features, "
            "'auto' uses available extended/nextgen columns when present."
        ),
    )
    parser.add_argument(
        "--missing-feature-policy",
        choices=["off", "warn", "error"],
        default="warn",
        help=(
            "Policy for missing/empty required feature columns. "
            "For --feature-profile=nextgen, set to 'error' to fail fast instead of zero-fill fallback."
        ),
    )
    parser.add_argument(
        "--nextgen-required-cols",
        choices=["none", "critical", "all"],
        default="critical",
        help=(
            "Required-column set used by --missing-feature-policy when --feature-profile=nextgen. "
            "'critical' checks key embedding/discrepancy/distortion columns; "
            "'all' checks the full nextgen schema."
        ),
    )
    parser.add_argument(
        "--class-balance-mode",
        choices=["none", "downsample", "upsample"],
        default="none",
        help="Optional row-level class balancing applied before train/val split.",
    )
    parser.add_argument(
        "--group-split-col",
        type=str,
        default="video_id",
        help="Optional grouping column for leakage-safe split (default: video_id). Use empty string to disable.",
    )
    parser.add_argument(
        "--group-balance",
        action="store_true",
        help="Apply domain+label inverse-frequency sample weighting when dataset column is available.",
    )
    parser.add_argument(
        "--generator-balance",
        action="store_true",
        help="Apply pseudo-generator/domain balancing weights to reduce single-generator shortcut learning.",
    )
    parser.add_argument(
        "--generator-balance-col",
        type=str,
        default="dataset",
        help="Column used for generator/domain balancing. Falls back to dataset+scenario key when unavailable.",
    )
    parser.add_argument(
        "--generator-balance-mode",
        choices=["inverse_freq", "effective_num"],
        default="effective_num",
        help="Weighting rule for generator/domain balancing.",
    )
    parser.add_argument(
        "--generator-balance-beta",
        type=float,
        default=0.999,
        help="Beta for effective-number generator balancing (used when --generator-balance-mode=effective_num).",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Use WeightedRandomSampler for train batches (uses computed sample weights).",
    )
    parser.add_argument(
        "--weight-application",
        choices=["auto", "loss", "sampler", "both", "none"],
        default="auto",
        help=(
            "How class/sample weights are applied. "
            "'auto' avoids double-weighting by using sampler OR loss weights; "
            "'none' disables both weighted loss and weighted sampling."
        ),
    )
    parser.add_argument(
        "--hard-negative-file",
        type=str,
        default=None,
        help="Optional jsonl/csv/tsv with hard negatives to upweight (real predicted fake).",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=2.0,
        help="Multiplicative weight for matched hard-negative training rows.",
    )
    parser.add_argument(
        "--hard-positive-file",
        type=str,
        default=None,
        help="Optional jsonl/csv/tsv with hard positives to upweight (fake predicted real).",
    )
    parser.add_argument(
        "--hard-positive-weight",
        type=float,
        default=2.0,
        help="Multiplicative weight for matched hard-positive training rows.",
    )
    parser.add_argument(
        "--scenario-focus",
        type=str,
        default="none",
        choices=["none", "video_only_fake", "audio_only_fake", "both_fake"],
        help=(
            "Optional scenario-aware upweighting target using video_fake/audio_fake labels. "
            "'video_only_fake' is useful when recall is weak on visual-only forgeries."
        ),
    )
    parser.add_argument(
        "--scenario-focus-weight",
        type=float,
        default=1.0,
        help="Multiplicative sample-weight factor for --scenario-focus rows.",
    )
    parser.add_argument(
        "--paired-rf-sampling",
        action="store_true",
        help=(
            "Enable paired real/fake sampling so each train batch contains balanced "
            "real-vs-fake examples."
        ),
    )
    parser.add_argument(
        "--paired-rf-source-col",
        type=str,
        default="pair_source_id",
        help=(
            "Optional grouping column for paired real/fake sampling. "
            "When both classes share a group value, pairs are drawn within-group."
        ),
    )
    parser.add_argument(
        "--gend-visual-adapt",
        action="store_true",
        help=(
            "Enable GenD-style visual adaptation that nudges fake visual embeddings "
            "toward paired real embeddings inside each train batch."
        ),
    )
    parser.add_argument(
        "--gend-adapt-strength",
        type=float,
        default=0.35,
        help="Max interpolation strength for GenD visual adaptation in [0,1].",
    )
    parser.add_argument(
        "--gend-adapt-prob",
        type=float,
        default=1.0,
        help="Per-batch probability of applying GenD visual adaptation in [0,1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for split and training reproducibility.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="robust_bal",
        choices=["auc", "robust_bal", "hybrid_robust"],
        help="Metric used for best-checkpoint selection (default: robust_bal).",
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for robustness metrics during selection.",
    )
    parser.add_argument(
        "--selection-threshold-mode",
        type=str,
        default="fixed",
        choices=["fixed", "best_f1", "target"],
        help=(
            "Threshold mode for robustness metrics/checkpoint constraints: "
            "'fixed' uses --selection-threshold, 'best_f1' uses per-epoch swept best_thr, "
            "'target' calibrates per epoch to satisfy classification targets."
        ),
    )
    parser.add_argument(
        "--target-acc",
        type=float,
        default=None,
        help="Optional target accuracy used when --selection-threshold-mode=target.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=None,
        help="Optional target precision used when --selection-threshold-mode=target.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Optional target recall used when --selection-threshold-mode=target.",
    )
    parser.add_argument(
        "--target-f1",
        type=float,
        default=None,
        help="Optional target F1 used when --selection-threshold-mode=target.",
    )
    parser.add_argument(
        "--target-spec",
        type=float,
        default=None,
        help="Optional target specificity used when --selection-threshold-mode=target.",
    )
    parser.add_argument(
        "--target-priority",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "precision", "recall", "balanced_acc"],
        help="Priority metric used to break ties among thresholds that meet targets.",
    )
    parser.add_argument(
        "--enable-temperature-calibration",
        action="store_true",
        help="Fit a post-hoc scalar temperature on validation probabilities and save it with model artifacts.",
    )
    parser.add_argument(
        "--temperature-min",
        type=float,
        default=0.5,
        help="Minimum temperature considered for post-hoc temperature scaling.",
    )
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=3.0,
        help="Maximum temperature considered for post-hoc temperature scaling.",
    )
    parser.add_argument(
        "--temperature-steps",
        type=int,
        default=61,
        help="Grid-search steps for post-hoc temperature scaling.",
    )
    parser.add_argument(
        "--min-domain-spec",
        type=float,
        default=0.0,
        help="Minimum worst-domain specificity required to accept a checkpoint.",
    )
    parser.add_argument(
        "--min-domain-rec",
        type=float,
        default=0.0,
        help="Minimum worst-domain recall required to accept a checkpoint.",
    )
    args = parser.parse_args()

    if args.auc_margin_loss_weight is not None:
        args.ranking_loss_weight = float(args.auc_margin_loss_weight)
    if args.auc_margin is not None:
        args.ranking_margin = float(args.auc_margin)
    if args.auc_margin_max_pairs is not None:
        args.ranking_max_pairs = int(args.auc_margin_max_pairs)
    if bool(args.joint_ce_auc_margin) and float(args.ranking_loss_weight) <= 0.0:
        args.ranking_loss_weight = 0.2
    if args.distortion_loss_weight is not None:
        args.causal_breach_loss_weight = float(args.distortion_loss_weight)
    if args.distortion_target is not None:
        args.causal_breach_target = str(args.distortion_target)
    if args.distortion_column is not None:
        args.causal_breach_column = str(args.distortion_column)
    if bool(args.ln_only_finetune):
        args.enable_av_layernorm_adapter = True
    if bool(args.stage0_avc_pretrain):
        args.stage0_real_only = True

    target_metrics = {}
    for arg_name, metric_key in [
        ("target_acc", "acc"),
        ("target_precision", "prec"),
        ("target_recall", "rec"),
        ("target_f1", "f1"),
        ("target_spec", "spec"),
    ]:
        raw_v = getattr(args, arg_name)
        if raw_v is None:
            continue
        if float(raw_v) < 0.0 or float(raw_v) > 1.0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be within [0, 1].")
        target_metrics[metric_key] = float(raw_v)

    if args.selection_threshold_mode == "target" and not target_metrics:
        raise ValueError(
            "--selection-threshold-mode=target requires at least one target metric "
            "(--target-acc/--target-precision/--target-recall/--target-f1/--target-spec)."
        )
    if target_metrics:
        print(f"Target metrics for threshold calibration/reporting: {target_metrics}")
    if float(args.hard_negative_weight) <= 0.0:
        raise ValueError("--hard-negative-weight must be > 0.")
    if float(args.hard_positive_weight) <= 0.0:
        raise ValueError("--hard-positive-weight must be > 0.")
    if float(args.scenario_focus_weight) <= 0.0:
        raise ValueError("--scenario-focus-weight must be > 0.")
    if args.focal_bce_mix is not None and (float(args.focal_bce_mix) < 0.0 or float(args.focal_bce_mix) > 1.0):
        raise ValueError("--focal-bce-mix must be within [0, 1].")
    if float(args.class_balance_beta) < 0.0 or float(args.class_balance_beta) >= 1.0:
        raise ValueError("--class-balance-beta must be in [0, 1).")
    if float(args.hyperspherical_reg_weight) < 0.0:
        raise ValueError("--hyperspherical-reg-weight must be >= 0.")
    if float(args.hyperspherical_margin) < 0.0 or float(args.hyperspherical_margin) > 1.0:
        raise ValueError("--hyperspherical-margin must be within [0, 1].")
    if float(args.generator_balance_beta) < 0.0 or float(args.generator_balance_beta) >= 1.0:
        raise ValueError("--generator-balance-beta must be in [0, 1).")
    if bool(args.paired_rf_sampling) and int(args.batch_size) < 2:
        raise ValueError("--paired-rf-sampling requires --batch-size >= 2.")
    if float(args.ranking_loss_weight) < 0.0:
        raise ValueError("--ranking-loss-weight must be >= 0.")
    if float(args.ranking_margin) < 0.0:
        raise ValueError("--ranking-margin must be >= 0.")
    if int(args.ranking_max_pairs) < 1:
        raise ValueError("--ranking-max-pairs must be >= 1.")
    if float(args.gend_adapt_strength) < 0.0 or float(args.gend_adapt_strength) > 1.0:
        raise ValueError("--gend-adapt-strength must be within [0, 1].")
    if float(args.gend_adapt_prob) < 0.0 or float(args.gend_adapt_prob) > 1.0:
        raise ValueError("--gend-adapt-prob must be within [0, 1].")
    if float(args.causal_breach_loss_weight) < 0.0:
        raise ValueError("--causal-breach-loss-weight must be >= 0.")
    if float(args.causal_breach_loss_weight) > 0.0 and str(args.causal_breach_target) == "none":
        raise ValueError(
            "--causal-breach-loss-weight > 0 requires --causal-breach-target "
            "to be 'column', 'heuristic', or 'artifact_heuristic'."
        )
    if bool(args.stage0_avc_pretrain):
        if int(args.stage0_epochs) < 1:
            raise ValueError("--stage0-epochs must be >= 1.")
        if int(args.stage0_patience) < 1:
            raise ValueError("--stage0-patience must be >= 1.")
        if float(args.stage0_lr) <= 0.0:
            raise ValueError("--stage0-lr must be > 0.")
        if float(args.stage0_temperature) <= 0.0:
            raise ValueError("--stage0-temperature must be > 0.")
        if float(args.stage0_lip_weight) < 0.0:
            raise ValueError("--stage0-lip-weight must be >= 0.")
    if bool(args.enable_temperature_calibration):
        if float(args.temperature_min) <= 0.0:
            raise ValueError("--temperature-min must be > 0.")
        if float(args.temperature_max) < float(args.temperature_min):
            raise ValueError("--temperature-max must be >= --temperature-min.")
        if int(args.temperature_steps) < 3:
            raise ValueError("--temperature-steps must be >= 3.")

    set_seed(args.seed)
    print(f"Using seed: {args.seed}")

    df = pd.read_csv(args.data)
    before_rows = len(df)
    df = filter_training_sources(df, args.train_source)
    after_source_rows = len(df)
    if after_source_rows == 0:
        raise RuntimeError("No rows available after source filtering.")
    print(
        f"Loaded {before_rows} rows from {args.data}; "
        f"using {after_source_rows} rows after --train-source={args.train_source} filter."
    )
    if args.class_balance_mode != "none":
        rows_before_balance = len(df)
        df = rebalance_by_label(df, mode=args.class_balance_mode, label_col="label", seed=args.seed)
        print(f"Rows after class balancing: {rows_before_balance} -> {len(df)}")

    _validate_feature_coverage(
        df=df,
        feature_profile=str(args.feature_profile),
        use_embeddings=bool(args.use_embeddings),
        required_mode=str(args.nextgen_required_cols),
        policy=str(args.missing_feature_policy),
    )

    av_feature_cols, phys_feature_cols = resolve_feature_columns(
        df.columns,
        use_embeddings=bool(args.use_embeddings),
        profile=args.feature_profile,
    )
    visual_adapt_indices = _resolve_gend_visual_indices(av_feature_cols)
    lip_feature_cols = resolve_lip_feature_columns(
        df.columns,
        enable_lip_stream=bool(args.enable_lip_stream),
    )
    av_feats = build_feature_matrix(df, av_feature_cols, name="AV features")
    phys_feats = build_feature_matrix(df, phys_feature_cols, name="Physical features")
    lip_feats = build_feature_matrix(df, lip_feature_cols, name="Lip stream features")
    print(
        f"Feature profile={args.feature_profile}: "
        f"av_dim={av_feats.shape[1]} phys_dim={phys_feats.shape[1]} "
        f"lip_dim={lip_feats.shape[1]}"
    )
    enable_gend_visual_adapt = bool(
        args.gend_visual_adapt
        and len(visual_adapt_indices) > 0
        and float(args.gend_adapt_strength) > 0.0
    )
    if args.gend_visual_adapt and not enable_gend_visual_adapt:
        print("GenD visual adaptation requested, but no compatible visual embedding columns were found.")
    if enable_gend_visual_adapt:
        active_visual_cols = [av_feature_cols[i] for i in visual_adapt_indices]
        print(
            "GenD visual adaptation enabled "
            f"(strength={float(args.gend_adapt_strength):.3f}, "
            f"prob={float(args.gend_adapt_prob):.3f}, "
            f"cols={active_visual_cols})."
        )
    labels = df["label"].values
    domain_labels = infer_domain_labels(df)
    pos_rate = float(np.mean(labels.astype(float)))
    print(f"Label prevalence: pos={pos_rate:.4f}, neg={1.0 - pos_rate:.4f}")
    if args.selection_threshold_mode in {"best_f1", "target"} and (pos_rate > 0.90 or pos_rate < 0.10):
        print(
            f"Warning: --selection-threshold-mode={args.selection_threshold_mode} with extreme "
            "class skew can collapse specificity/recall trade-offs. Consider mode='fixed' "
            "or explicit --target-* constraints."
        )

    idx_train, idx_val = split_train_val_indices(
        df,
        val_split=args.val_split,
        seed=args.seed,
        domain_labels=domain_labels,
        group_col=args.group_split_col,
    )
    train_df = df.iloc[idx_train].copy().reset_index(drop=True)
    val_df = df.iloc[idx_val].copy().reset_index(drop=True)

    X_av_train = av_feats[idx_train]
    X_av_val = av_feats[idx_val]
    X_phys_train = phys_feats[idx_train]
    X_phys_val = phys_feats[idx_val]
    X_lip_train = lip_feats[idx_train] if lip_feats.shape[1] > 0 else None
    X_lip_val = lip_feats[idx_val] if lip_feats.shape[1] > 0 else None
    y_train_arr = labels[idx_train]
    y_val_arr = labels[idx_val]
    val_domains = infer_domain_labels(val_df)

    causal_target_train_arr = None
    causal_target_val_arr = None
    causal_target_source = "disabled"
    if float(args.causal_breach_loss_weight) > 0.0:
        causal_target_train_arr, causal_target_source = infer_causal_breach_targets(
            train_df,
            mode=str(args.causal_breach_target),
            column=str(args.causal_breach_column),
        )
        causal_target_val_arr, _ = infer_causal_breach_targets(
            val_df,
            mode=str(args.causal_breach_target),
            column=str(args.causal_breach_column),
        )
        if causal_target_train_arr is None:
            raise RuntimeError("Causal-breach targets could not be resolved for training.")
        valid_train = float(np.mean(causal_target_train_arr >= 0.0))
        valid_val = float(np.mean(causal_target_val_arr >= 0.0))
        print(
            "Causal breach supervision enabled: "
            f"source={causal_target_source} "
            f"weight={float(args.causal_breach_loss_weight):.3f} "
            f"valid_train={valid_train:.3f} valid_val={valid_val:.3f}"
        )

    # Auxiliary task labels (unknown -> -1). Keeps compatibility with DFDC rows.
    def _extract_aux(arr_df, col):
        if col not in arr_df.columns:
            return np.full(len(arr_df), -1.0, dtype=np.float32)
        vals = (
            pd.to_numeric(arr_df[col], errors="coerce")
            .fillna(-1.0)
            .astype(np.float32)
            .to_numpy(copy=True)
        )
        vals[(vals < 0.0) | (vals > 1.0)] = -1.0
        return vals

    video_train_arr = _extract_aux(train_df, "video_fake")
    video_val_arr = _extract_aux(val_df, "video_fake")
    audio_train_arr = _extract_aux(train_df, "audio_fake")
    audio_val_arr = _extract_aux(val_df, "audio_fake")
    aux_available = bool(
        np.any(video_train_arr >= 0.0)
        or np.any(audio_train_arr >= 0.0)
        or np.any(video_val_arr >= 0.0)
        or np.any(audio_val_arr >= 0.0)
    )
    enable_multitask = bool(args.enable_multitask and aux_available)
    if args.enable_multitask and not enable_multitask:
        print("Multi-task requested, but no valid video_fake/audio_fake labels found; disabling auxiliary heads.")
    if enable_multitask:
        print(
            "Multi-task enabled "
            f"(aux_weight={float(args.multitask_weight):.3f})."
        )
    if float(args.ranking_loss_weight) > 0.0:
        print(
            "AUC-margin ranking loss enabled "
            f"(weight={float(args.ranking_loss_weight):.3f}, "
            f"margin={float(args.ranking_margin):.3f}, "
            f"max_pairs={int(args.ranking_max_pairs)})."
        )
        if bool(args.joint_ce_auc_margin):
            print("Joint CE + AUC-margin objective enabled.")
    if args.focal_bce_mix is not None:
        print(
            "Mixed focal/BCE objective enabled "
            f"(focal_mix={float(args.focal_bce_mix):.3f}, bce_mix={1.0 - float(args.focal_bce_mix):.3f})."
        )
    if float(args.hyperspherical_reg_weight) > 0.0:
        print(
            "Hyperspherical regularization enabled "
            f"(weight={float(args.hyperspherical_reg_weight):.3f}, margin={float(args.hyperspherical_margin):.3f})."
        )
    if bool(args.enable_av_layernorm_adapter):
        print("AV LayerNorm adapter enabled.")
    enable_lip_stream = bool(args.enable_lip_stream and X_lip_train is not None and X_lip_train.shape[1] > 0)
    if args.enable_lip_stream and not enable_lip_stream:
        print("Lip stream requested, but no lip features available; disabling lip stream.")
    if enable_lip_stream:
        print(f"Lip stream enabled (lip_dim={int(X_lip_train.shape[1])}).")

    enable_causal_breach_head = bool(
        float(args.causal_breach_loss_weight) > 0.0
        and causal_target_train_arr is not None
        and np.any(causal_target_train_arr >= 0.0)
    )
    if float(args.causal_breach_loss_weight) > 0.0 and not enable_causal_breach_head:
        print("Causal breach loss requested, but no valid targets found; disabling causal breach head/loss.")
    if enable_causal_breach_head:
        print(
            "Causal breach head enabled "
            f"(weight={float(args.causal_breach_loss_weight):.3f}, source={causal_target_source})."
        )

    scaler = None
    if args.use_scaler:
        scaler = {
            "av": StandardScaler().fit(X_av_train),
            "phys": StandardScaler().fit(X_phys_train),
        }
        X_av_train = scaler["av"].transform(X_av_train)
        X_av_val = scaler["av"].transform(X_av_val)
        X_phys_train = scaler["phys"].transform(X_phys_train)
        X_phys_val = scaler["phys"].transform(X_phys_val)
        if enable_lip_stream:
            scaler["lip"] = StandardScaler().fit(X_lip_train)
            X_lip_train = scaler["lip"].transform(X_lip_train)
            X_lip_val = scaler["lip"].transform(X_lip_val)

    X_av_train = torch.tensor(X_av_train, dtype=torch.float32)
    X_phys_train = torch.tensor(X_phys_train, dtype=torch.float32)
    X_lip_train_t = torch.tensor(X_lip_train, dtype=torch.float32) if enable_lip_stream else None
    y_train = torch.tensor(y_train_arr, dtype=torch.float32).unsqueeze(1)
    y_video_train = torch.tensor(video_train_arr, dtype=torch.float32).unsqueeze(1)
    y_audio_train = torch.tensor(audio_train_arr, dtype=torch.float32).unsqueeze(1)
    y_causal_breach_train = (
        torch.tensor(causal_target_train_arr, dtype=torch.float32).unsqueeze(1)
        if enable_causal_breach_head
        else None
    )

    X_av_val = torch.tensor(X_av_val, dtype=torch.float32)
    X_phys_val = torch.tensor(X_phys_val, dtype=torch.float32)
    X_lip_val_t = torch.tensor(X_lip_val, dtype=torch.float32) if enable_lip_stream else None
    y_val = torch.tensor(y_val_arr, dtype=torch.float32).unsqueeze(1)
    y_video_val = torch.tensor(video_val_arr, dtype=torch.float32).unsqueeze(1)
    y_audio_val = torch.tensor(audio_val_arr, dtype=torch.float32).unsqueeze(1)
    y_causal_breach_val = (
        torch.tensor(causal_target_val_arr, dtype=torch.float32).unsqueeze(1)
        if enable_causal_breach_head
        else None
    )

    weight_mode = (args.weight_application or "auto").strip().lower()
    if weight_mode == "auto":
        apply_sampler_weights = bool(args.use_weighted_sampler)
        apply_loss_weights = not apply_sampler_weights
        resolved_mode = "sampler" if apply_sampler_weights else "loss"
        print(f"Weight application (auto) resolved to: {resolved_mode}.")
    else:
        apply_loss_weights = weight_mode in {"loss", "both"}
        apply_sampler_weights = weight_mode in {"sampler", "both"}
        if args.use_weighted_sampler and not apply_sampler_weights:
            print(
                f"Ignoring --use-weighted-sampler because --weight-application={weight_mode}."
            )
        print(f"Weight application mode: loss={apply_loss_weights} sampler={apply_sampler_weights}")

    sample_weights_np = None
    if len(np.unique(y_train_arr)) > 1:
        # Base class balancing
        class_counts = np.bincount(y_train_arr.astype(int))
        class_weights = class_counts.sum() / np.maximum(class_counts, 1)
        sample_weights_np = np.array([class_weights[int(l)] for l in y_train_arr], dtype=np.float32)

        train_key_mode, train_keys = _resolve_error_lookup_paths(train_df)
        train_lookup_ready = train_key_mode in {"path", "video_id"}

        # Optional domain+label balancing
        if args.group_balance and "dataset" in train_df.columns:
            grp = (
                train_df["dataset"].astype(str).str.lower().fillna("unknown")
                + ":"
                + pd.Series(y_train_arr, index=train_df.index).astype(int).astype(str)
            )
            grp_counts = grp.value_counts().to_dict()
            grp_weights = {k: len(grp) / max(v, 1) for k, v in grp_counts.items()}
            sample_weights_np *= grp.map(grp_weights).to_numpy(dtype=np.float32)
            print(f"Applied group balancing across {len(grp_weights)} dataset-label groups.")

        # Optional generator/domain balancing (D3-style many-generator robustness).
        if bool(args.generator_balance):
            gen_groups, gen_status = _resolve_generator_groups(train_df, str(args.generator_balance_col))
            gen_w = _generator_balance_weights(
                gen_groups,
                mode=str(args.generator_balance_mode),
                beta=float(args.generator_balance_beta),
            )
            if gen_w is None:
                print(f"Generator balance skipped: group source {gen_status}.")
            else:
                sample_weights_np *= gen_w.astype(np.float32, copy=False)
                print(
                    "Applied generator/domain balancing "
                    f"(source={gen_status}, mode={args.generator_balance_mode})."
                )

        # Optional hard-negative upweighting
        if args.hard_negative_file:
            hn_paths = load_hard_negative_paths(args.hard_negative_file)
            hn_lookup = _normalize_lookup_paths(hn_paths, train_key_mode)
            hn_mask = train_keys.isin(hn_lookup).to_numpy() & (y_train_arr.astype(int) == 0)
            hn_count = int(np.sum(hn_mask))
            if hn_count > 0:
                sample_weights_np[hn_mask] *= float(args.hard_negative_weight)
            print(f"Applied hard-negative weight to {hn_count} training rows.")
            if not train_lookup_ready:
                print("Hard-negative lookup fallback: missing path/video_id columns in train CSV.")

        # Optional hard-positive upweighting
        if args.hard_positive_file:
            hp_paths = load_hard_positive_paths(args.hard_positive_file)
            hp_lookup = _normalize_lookup_paths(hp_paths, train_key_mode)
            hp_mask = train_keys.isin(hp_lookup).to_numpy() & (y_train_arr.astype(int) == 1)
            hp_count = int(np.sum(hp_mask))
            if hp_count > 0:
                sample_weights_np[hp_mask] *= float(args.hard_positive_weight)
            print(f"Applied hard-positive weight to {hp_count} training rows.")
            if not train_lookup_ready:
                print("Hard-positive lookup fallback: missing path/video_id columns in train CSV.")

        # Optional scenario-aware upweighting (e.g., video_only_fake)
        if str(args.scenario_focus) != "none":
            scen_mask, scen_status = _scenario_focus_mask(train_df, str(args.scenario_focus))
            if scen_status == "missing_video_audio_cols":
                print("Scenario focus skipped: missing video_fake/audio_fake columns.")
            else:
                scen_mask = scen_mask & (y_train_arr.astype(int) == 1)
                scen_count = int(np.sum(scen_mask))
                if scen_count > 0:
                    sample_weights_np[scen_mask] *= float(args.scenario_focus_weight)
                print(
                    "Applied scenario-focus weight to "
                    f"{scen_count} rows (scenario={args.scenario_focus}, "
                    f"weight={float(args.scenario_focus_weight):.3f})."
                )

        # Normalize mean weight for stable loss scale
        sample_weights_np = sample_weights_np / max(float(np.mean(sample_weights_np)), 1e-6)
    else:
        print("Sample weighting skipped: single class in training split.")

    class_balanced_weights = None
    if bool(args.class_balanced_loss):
        class_balanced_weights = _effective_num_class_weights(
            y_train_arr.astype(int),
            beta=float(args.class_balance_beta),
        )
        print(
            "Class-balanced effective-number loss enabled "
            f"(beta={float(args.class_balance_beta):.6f}, "
            f"w_neg={class_balanced_weights[0]:.4f}, "
            f"w_pos={class_balanced_weights[1]:.4f})."
        )

    sample_weights = None
    use_weights = False
    if sample_weights_np is not None and apply_loss_weights:
        sample_weights = torch.tensor(sample_weights_np, dtype=torch.float32).unsqueeze(1)
        use_weights = True

    sampler = None
    paired_batch_sampler = None
    enable_paired_rf_sampling = False
    paired_group_count = 0
    paired_groups = None
    paired_group_status = "disabled"
    if bool(args.paired_rf_sampling):
        paired_groups, paired_group_status = _resolve_pair_groups(
            train_df,
            source_col=str(args.paired_rf_source_col),
        )
    if bool(args.paired_rf_sampling):
        if len(np.unique(y_train_arr.astype(int))) < 2:
            print("Paired real/fake sampling requested, but training split has a single class; disabling.")
        else:
            paired_sw = sample_weights_np if apply_sampler_weights else None
            paired_batch_sampler = PairedRealFakeBatchSampler(
                labels=y_train_arr.astype(int),
                batch_size=int(args.batch_size),
                seed=int(args.seed),
                sample_weights=paired_sw,
                pair_groups=paired_groups,
            )
            enable_paired_rf_sampling = True
            if paired_batch_sampler.paired_group_keys is not None:
                paired_group_count = int(paired_batch_sampler.paired_group_keys.size)
            print(
                "Using paired real/fake batch sampler for train batches "
                f"(source={paired_group_status}, paired_groups={paired_group_count})."
            )
    elif sample_weights_np is not None and apply_sampler_weights:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights_np, dtype=torch.double),
            num_samples=len(sample_weights_np),
            replacement=True,
        )
        print("Using WeightedRandomSampler for train batches.")
    elif apply_sampler_weights:
        print("Weighted sampler requested but unavailable (no sample weights computed).")

    train_dataset_weights = sample_weights if use_weights else None
    train_loader = build_loaders(
        X_av_train,
        X_phys_train,
        y_train,
        args.batch_size,
        shuffle=True,
        X_lip=(X_lip_train_t if enable_lip_stream else None),
        y_video=(y_video_train if enable_multitask else None),
        y_audio=(y_audio_train if enable_multitask else None),
        y_causal_breach=(y_causal_breach_train if enable_causal_breach_head else None),
        weights=train_dataset_weights,
        sampler=sampler,
        batch_sampler=paired_batch_sampler,
    )

    val_loader = build_loaders(
        X_av_val,
        X_phys_val,
        y_val,
        args.batch_size,
        shuffle=False,
        X_lip=(X_lip_val_t if enable_lip_stream else None),
        y_video=(y_video_val if enable_multitask else None),
        y_audio=(y_audio_val if enable_multitask else None),
        y_causal_breach=(y_causal_breach_val if enable_causal_breach_head else None),
        weights=None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_v2 = (
        bool(args.use_embeddings)
        or enable_lip_stream
        or enable_causal_breach_head
        or int(X_av_train.shape[1]) != 3
        or int(X_phys_train.shape[1]) != 2
    )
    if use_v2:
        model = CausalFusionNetworkV2(
            av_dim=X_av_train.shape[1],
            phys_dim=X_phys_train.shape[1],
            enable_multitask=enable_multitask,
            lip_dim=(int(X_lip_train_t.shape[1]) if enable_lip_stream else 0),
            enable_causal_breach_head=enable_causal_breach_head,
            enable_av_input_layernorm=bool(args.enable_av_layernorm_adapter),
        ).to(device)
    else:
        model = CausalFusionNetwork(
            enable_multitask=enable_multitask,
            enable_causal_breach_head=enable_causal_breach_head,
            enable_av_input_layernorm=bool(args.enable_av_layernorm_adapter),
        ).to(device)

    if args.init_model:
        init_path = Path(args.init_model)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-model not found: {init_path}")
        init_state = torch.load(str(init_path), map_location=device)
        incompatible = model.load_state_dict(init_state, strict=bool(args.init_strict))
        if hasattr(incompatible, "missing_keys") and hasattr(incompatible, "unexpected_keys"):
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            print(
                "Loaded init checkpoint: "
                f"{init_path} "
                f"(strict={bool(args.init_strict)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if missing:
                print("  missing_keys:", missing[:10])
            if unexpected:
                print("  unexpected_keys:", unexpected[:10])
        else:
            print(f"Loaded init checkpoint: {init_path} (strict={bool(args.init_strict)})")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "cfn_emb.pth" if args.use_embeddings else "cfn.pth")
    scaler_path = os.path.join(args.model_dir, "cfn_scaler.pkl")
    threshold_report_path = os.path.join(args.model_dir, "cfn_threshold_report.json")
    temperature_path = os.path.join(args.model_dir, "cfn_temperature.json")

    stage0_report = {
        "enabled": bool(args.stage0_avc_pretrain),
        "status": "disabled",
        "epochs": int(args.stage0_epochs),
        "patience": int(args.stage0_patience),
        "lr": float(args.stage0_lr),
        "temperature": float(args.stage0_temperature),
        "lip_weight": float(args.stage0_lip_weight),
        "real_only": bool(args.stage0_real_only),
        "best_epoch": None,
        "best_val_loss": None,
        "model_path": None,
    }
    if bool(args.stage0_avc_pretrain):
        stage0_train_mask = (y_train_arr.astype(int) == 0) if bool(args.stage0_real_only) else np.ones_like(y_train_arr, dtype=bool)
        stage0_val_mask = (y_val_arr.astype(int) == 0) if bool(args.stage0_real_only) else np.ones_like(y_val_arr, dtype=bool)
        idx_stage0_train = np.where(stage0_train_mask)[0]
        idx_stage0_val = np.where(stage0_val_mask)[0]

        if idx_stage0_train.size < 2:
            stage0_report["status"] = "skipped_insufficient_rows"
            print("Stage-0 AV correspondence pretraining skipped: insufficient eligible rows.")
        else:
            stage0_tensors_train = [X_av_train[idx_stage0_train], X_phys_train[idx_stage0_train]]
            if enable_lip_stream and X_lip_train_t is not None:
                stage0_tensors_train.append(X_lip_train_t[idx_stage0_train])
            stage0_train_loader = DataLoader(
                TensorDataset(*stage0_tensors_train),
                batch_size=int(args.batch_size),
                shuffle=True,
            )

            stage0_val_loader = None
            if idx_stage0_val.size >= 2:
                stage0_tensors_val = [X_av_val[idx_stage0_val], X_phys_val[idx_stage0_val]]
                if enable_lip_stream and X_lip_val_t is not None:
                    stage0_tensors_val.append(X_lip_val_t[idx_stage0_val])
                stage0_val_loader = DataLoader(
                    TensorDataset(*stage0_tensors_val),
                    batch_size=int(args.batch_size),
                    shuffle=False,
                )

            stage0_params = [p for p in model.parameters() if p.requires_grad]
            if not stage0_params:
                stage0_report["status"] = "skipped_no_trainable_params"
                print("Stage-0 AV correspondence pretraining skipped: no trainable parameters.")
            else:
                stage0_opt = torch.optim.AdamW(
                    stage0_params,
                    lr=float(args.stage0_lr),
                    weight_decay=float(args.weight_decay),
                )
                stage0_best_loss = float("inf")
                stage0_best_epoch = 0
                stage0_best_state = None
                stage0_no_improve = 0
                print(
                    "Running Stage-0 AV correspondence pretraining "
                    f"(rows_train={idx_stage0_train.size}, rows_val={idx_stage0_val.size}, "
                    f"epochs={int(args.stage0_epochs)}, patience={int(args.stage0_patience)}, "
                    f"real_only={bool(args.stage0_real_only)})."
                )
                for s_epoch in range(int(args.stage0_epochs)):
                    s_train_loss = _stage0_av_corr_epoch(
                        model,
                        stage0_train_loader,
                        device,
                        optimizer=stage0_opt,
                        temperature=float(args.stage0_temperature),
                        lip_weight=float(args.stage0_lip_weight),
                    )
                    if stage0_val_loader is not None:
                        s_val_loss = _stage0_av_corr_epoch(
                            model,
                            stage0_val_loader,
                            device,
                            optimizer=None,
                            temperature=float(args.stage0_temperature),
                            lip_weight=float(args.stage0_lip_weight),
                        )
                    else:
                        s_val_loss = float(s_train_loss)
                    print(
                        f"[Stage0] epoch={s_epoch + 1:02d} "
                        f"train_loss={float(s_train_loss):.4f} val_loss={float(s_val_loss):.4f}"
                    )
                    if float(s_val_loss) < stage0_best_loss:
                        stage0_best_loss = float(s_val_loss)
                        stage0_best_epoch = int(s_epoch + 1)
                        stage0_best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        stage0_no_improve = 0
                    else:
                        stage0_no_improve += 1
                    if stage0_no_improve >= int(args.stage0_patience):
                        print("[Stage0] early stopping triggered.")
                        break

                if stage0_best_state is not None:
                    model.load_state_dict(stage0_best_state)
                    stage0_ckpt_path = os.path.join(
                        args.model_dir,
                        "cfn_stage0_pretrain_emb.pth" if args.use_embeddings else "cfn_stage0_pretrain.pth",
                    )
                    torch.save(model.state_dict(), stage0_ckpt_path)
                    stage0_report.update(
                        {
                            "status": "ok",
                            "best_epoch": int(stage0_best_epoch),
                            "best_val_loss": float(stage0_best_loss),
                            "model_path": str(stage0_ckpt_path),
                        }
                    )
                    print(
                        "Stage-0 AV correspondence pretraining complete "
                        f"(best_epoch={int(stage0_best_epoch)}, best_val_loss={float(stage0_best_loss):.4f})."
                    )
                else:
                    stage0_report["status"] = "failed_no_checkpoint"
                    print("Stage-0 AV correspondence pretraining failed to produce a checkpoint.")

    ln_only_trainable_params = None
    if bool(args.ln_only_finetune):
        ln_only_trainable_params = _configure_ln_only_finetune(
            model,
            unfreeze_heads=bool(args.ln_only_unfreeze_heads),
        )
        print(
            "LN-only fine-tune enabled "
            f"(unfreeze_heads={bool(args.ln_only_unfreeze_heads)}, "
            f"trainable_params={int(ln_only_trainable_params)})."
        )

    # Keep per-sample losses available for weighting and auxiliary terms.
    criterion = torch.nn.BCELoss(reduction="none")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if not optim_params:
        raise RuntimeError("No trainable parameters remain for supervised training.")
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auc = -1.0
    best_selection_score = -1.0
    best_unconstrained_score = -1.0
    best_unconstrained_epoch = -1
    best_unconstrained_state = None
    best_checkpoint_report = None
    best_unconstrained_report = None
    last_epoch_report = None
    saved_checkpoint = False
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        use_causal = args.causal_weight > 0
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_weights=use_weights,
            use_causal=use_causal,
            causal_weight=args.causal_weight,
            loss_type=args.loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            focal_bce_mix=args.focal_bce_mix,
            enable_multitask=enable_multitask,
            multitask_weight=args.multitask_weight,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_margin=args.ranking_margin,
            ranking_max_pairs=args.ranking_max_pairs,
            class_balanced_weights=class_balanced_weights,
            hyperspherical_reg_weight=args.hyperspherical_reg_weight,
            hyperspherical_margin=args.hyperspherical_margin,
            has_lip=enable_lip_stream,
            has_aux=enable_multitask,
            has_causal_breach_target=enable_causal_breach_head,
            has_weight=(train_dataset_weights is not None),
            causal_breach_loss_weight=args.causal_breach_loss_weight,
            gend_visual_adapt=enable_gend_visual_adapt,
            gend_adapt_strength=args.gend_adapt_strength,
            gend_adapt_prob=args.gend_adapt_prob,
            visual_adapt_indices=visual_adapt_indices,
        )
        val_loss, val_acc, val_auc, val_labels, val_probs = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_causal=use_causal,
            causal_weight=args.causal_weight,
            loss_type=args.loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            focal_bce_mix=args.focal_bce_mix,
            enable_multitask=enable_multitask,
            multitask_weight=args.multitask_weight,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_margin=args.ranking_margin,
            ranking_max_pairs=args.ranking_max_pairs,
            class_balanced_weights=class_balanced_weights,
            hyperspherical_reg_weight=args.hyperspherical_reg_weight,
            hyperspherical_margin=args.hyperspherical_margin,
            has_lip=enable_lip_stream,
            has_aux=enable_multitask,
            has_causal_breach_target=enable_causal_breach_head,
            has_weight=False,
            causal_breach_loss_weight=args.causal_breach_loss_weight,
        )
        sweep = threshold_sweep(val_labels, val_probs)
        threshold_cal = None
        selection_threshold = float(args.selection_threshold)
        if args.selection_threshold_mode == "best_f1":
            selection_threshold = float(sweep["best_thr"])
        elif args.selection_threshold_mode == "target":
            threshold_cal = calibrate_threshold_to_targets(
                val_labels,
                val_probs,
                targets=target_metrics,
                priority=args.target_priority,
            )
            selection_threshold = float(threshold_cal["threshold"])
        domain_metrics = compute_domain_metrics(
            val_labels,
            val_probs,
            val_domains,
            threshold=selection_threshold,
        )
        overall_sel = domain_metrics["overall"]
        target_shortfall = None
        targets_ok = None
        if target_metrics:
            target_shortfall = float(
                sum(
                    max(0.0, float(tv) - float(overall_sel.get(k, 0.0)))
                    for k, tv in target_metrics.items()
                )
            )
            targets_ok = bool(target_shortfall <= 1e-12)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_auc)

        if args.selection_metric == "auc":
            selection_score = float(val_auc)
        elif args.selection_metric == "hybrid_robust":
            selection_score = (
                0.50 * float(domain_metrics["worst_bal_acc"])
                + 0.30 * float(domain_metrics["macro_bal_acc"])
                + 0.20 * float(domain_metrics["overall"]["f1"])
            )
        else:
            selection_score = float(domain_metrics["worst_bal_acc"])

        meets_domain_constraints = (
            float(domain_metrics["worst_spec"]) >= float(args.min_domain_spec)
            and float(domain_metrics["worst_rec"]) >= float(args.min_domain_rec)
        )

        target_text = ""
        if target_metrics:
            target_text = (
                f" targets_ok={'Y' if targets_ok else 'N'}"
                f" target_gap={target_shortfall:.3f}"
            )
            if threshold_cal is not None:
                target_text += f" feasible_thr={int(threshold_cal['num_feasible'])}"

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} "
            f"val_auc={val_auc:.3f} "
            f"pr_auc={sweep['pr_auc']:.3f} "
            f"best_f1={sweep['best_f1']:.3f} "
            f"best_thr={sweep['best_thr']:.3f} "
            f"sel_thr={selection_threshold:.3f} "
            f"sel_prec={overall_sel['prec']:.3f} "
            f"sel_rec={overall_sel['rec']:.3f} "
            f"sel_f1={overall_sel['f1']:.3f} "
            f"worst_bal={domain_metrics['worst_bal_acc']:.3f} "
            f"worst_spec={domain_metrics['worst_spec']:.3f} "
            f"worst_rec={domain_metrics['worst_rec']:.3f} "
            f"sel_score={selection_score:.3f}"
            f"{target_text}"
        )

        threshold_cal_report = None
        if threshold_cal is not None:
            threshold_cal_report = {
                "threshold": float(threshold_cal["threshold"]),
                "meets_all_targets": bool(threshold_cal["meets_all_targets"]),
                "target_shortfall": float(threshold_cal["target_shortfall"]),
                "priority_metric": str(threshold_cal["priority_metric"]),
                "num_feasible": int(threshold_cal["num_feasible"]),
                "metrics": {k: float(v) for k, v in threshold_cal["metrics"].items()},
            }
        epoch_report = {
            "epoch": int(epoch + 1),
            "selection_threshold_mode": str(args.selection_threshold_mode),
            "selection_threshold": float(selection_threshold),
            "selection_metric": str(args.selection_metric),
            "selection_score": float(selection_score),
            "val_auc": float(val_auc),
            "pr_auc": float(sweep["pr_auc"]),
            "best_f1": float(sweep["best_f1"]),
            "best_f1_threshold": float(sweep["best_thr"]),
            "overall_at_selection_threshold": {
                k: float(v) for k, v in overall_sel.items()
            },
            "worst_bal_acc": float(domain_metrics["worst_bal_acc"]),
            "worst_spec": float(domain_metrics["worst_spec"]),
            "worst_rec": float(domain_metrics["worst_rec"]),
            "target_metrics": {k: float(v) for k, v in target_metrics.items()},
            "targets_met": (None if targets_ok is None else bool(targets_ok)),
            "target_shortfall": (None if target_shortfall is None else float(target_shortfall)),
            "threshold_calibration": threshold_cal_report,
        }
        last_epoch_report = epoch_report

        if val_auc > best_auc:
            best_auc = val_auc

        if selection_score > best_unconstrained_score:
            best_unconstrained_score = selection_score
            best_unconstrained_epoch = epoch + 1
            best_unconstrained_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_unconstrained_report = epoch_report

        improved = selection_score > best_selection_score
        if improved and meets_domain_constraints:
            best_selection_score = selection_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            best_checkpoint_report = epoch_report
            saved_checkpoint = True
            if scaler is not None:
                joblib.dump(scaler, scaler_path)
        elif improved:
            epochs_no_improve += 1
            failed = []
            if float(domain_metrics["worst_spec"]) < float(args.min_domain_spec):
                failed.append(
                    f"worst_spec={domain_metrics['worst_spec']:.3f}<{args.min_domain_spec:.3f}"
                )
            if float(domain_metrics["worst_rec"]) < float(args.min_domain_rec):
                failed.append(
                    f"worst_rec={domain_metrics['worst_rec']:.3f}<{args.min_domain_rec:.3f}"
                )
            print(
                f"Checkpoint skipped (constraints): "
                + (" and ".join(failed) if failed else "constraint check failed")
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    print("Best AUC:", best_auc)
    print("Best selection score:", best_selection_score)
    if not saved_checkpoint:
        if best_unconstrained_state is not None:
            print(
                "Warning: no checkpoint met selection constraints; "
                f"saving best unconstrained epoch {best_unconstrained_epoch}."
            )
            model.load_state_dict(best_unconstrained_state)
        else:
            print("Warning: no checkpoint met selection constraints; saving last epoch weights.")
        torch.save(model.state_dict(), model_path)
        if scaler is not None:
            joblib.dump(scaler, scaler_path)

    temperature_report = {
        "enabled": bool(args.enable_temperature_calibration),
        "status": "disabled",
        "temperature": 1.0,
        "nll_before": None,
        "nll_after": None,
        "path": None,
    }
    if bool(args.enable_temperature_calibration):
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, _, _, t_labels, t_probs = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_causal=(args.causal_weight > 0),
            causal_weight=args.causal_weight,
            loss_type=args.loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            focal_bce_mix=args.focal_bce_mix,
            enable_multitask=enable_multitask,
            multitask_weight=args.multitask_weight,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_margin=args.ranking_margin,
            ranking_max_pairs=args.ranking_max_pairs,
            class_balanced_weights=class_balanced_weights,
            hyperspherical_reg_weight=args.hyperspherical_reg_weight,
            hyperspherical_margin=args.hyperspherical_margin,
            has_lip=enable_lip_stream,
            has_aux=enable_multitask,
            has_causal_breach_target=enable_causal_breach_head,
            has_weight=False,
            causal_breach_loss_weight=args.causal_breach_loss_weight,
        )
        fit = fit_temperature_scaler(
            t_labels,
            t_probs,
            t_min=float(args.temperature_min),
            t_max=float(args.temperature_max),
            num_steps=int(args.temperature_steps),
        )
        temperature_report.update(
            {
                "status": "ok",
                "temperature": float(fit["temperature"]),
                "nll_before": (
                    None if not np.isfinite(float(fit["nll_before"])) else float(fit["nll_before"])
                ),
                "nll_after": (
                    None if not np.isfinite(float(fit["nll_after"])) else float(fit["nll_after"])
                ),
                "path": str(temperature_path),
            }
        )
        with open(temperature_path, "w") as fp:
            json.dump(temperature_report, fp, indent=2)
        print(
            "✔ Temperature calibration saved "
            f"(T={float(fit['temperature']):.4f}, "
            f"NLL {float(fit['nll_before']):.4f}->{float(fit['nll_after']):.4f}) "
            f"to {temperature_path}"
        )

    chosen_report = best_checkpoint_report if saved_checkpoint else best_unconstrained_report
    if chosen_report is None:
        chosen_report = last_epoch_report
    if chosen_report is not None:
        report_payload = {
            "selection_metric": str(args.selection_metric),
            "selection_threshold_mode": str(args.selection_threshold_mode),
            "target_metrics": {k: float(v) for k, v in target_metrics.items()},
            "best_checkpoint_met_constraints": bool(saved_checkpoint),
            "stage0_avc_pretrain": stage0_report,
            "temperature_calibration": temperature_report,
            "training_objective": {
                "loss": str(args.loss),
                "focal_alpha": float(args.focal_alpha),
                "focal_gamma": float(args.focal_gamma),
                "focal_bce_mix": (None if args.focal_bce_mix is None else float(args.focal_bce_mix)),
                "causal_weight": float(args.causal_weight),
                "requested_multitask": bool(args.enable_multitask),
                "enable_multitask": bool(enable_multitask),
                "multitask_weight": float(args.multitask_weight),
                "joint_ce_auc_margin": bool(args.joint_ce_auc_margin),
                "ranking_loss_weight": float(args.ranking_loss_weight),
                "ranking_margin": float(args.ranking_margin),
                "ranking_max_pairs": int(args.ranking_max_pairs),
                "auc_margin_loss_weight": float(args.ranking_loss_weight),
                "auc_margin": float(args.ranking_margin),
                "auc_margin_max_pairs": int(args.ranking_max_pairs),
                "class_balanced_loss": bool(args.class_balanced_loss),
                "class_balance_beta": float(args.class_balance_beta),
                "class_balanced_weights": (
                    None
                    if class_balanced_weights is None
                    else {
                        "neg": float(class_balanced_weights[0]),
                        "pos": float(class_balanced_weights[1]),
                    }
                ),
                "hyperspherical_reg_weight": float(args.hyperspherical_reg_weight),
                "hyperspherical_margin": float(args.hyperspherical_margin),
                "enable_av_layernorm_adapter": bool(args.enable_av_layernorm_adapter),
                "ln_only_finetune": bool(args.ln_only_finetune),
                "ln_only_unfreeze_heads": bool(args.ln_only_unfreeze_heads),
                "ln_only_trainable_params": (
                    None if ln_only_trainable_params is None else int(ln_only_trainable_params)
                ),
                "enable_lip_stream": bool(enable_lip_stream),
                "lip_feature_count": int(X_lip_train_t.shape[1]) if enable_lip_stream else 0,
                "causal_breach_loss_weight": float(args.causal_breach_loss_weight),
                "causal_breach_target_mode": str(args.causal_breach_target),
                "causal_breach_target_column": str(args.causal_breach_column),
                "distortion_loss_weight": float(args.causal_breach_loss_weight),
                "distortion_target_mode": str(args.causal_breach_target),
                "distortion_target_column": str(args.causal_breach_column),
                "causal_breach_target_source": str(causal_target_source),
                "enable_causal_breach_head": bool(enable_causal_breach_head),
                "weight_application_mode": str(weight_mode),
                "use_loss_weights": bool(use_weights),
                "use_weighted_sampler": bool(sampler is not None),
                "generator_balance": bool(args.generator_balance),
                "generator_balance_col": str(args.generator_balance_col),
                "generator_balance_mode": str(args.generator_balance_mode),
                "generator_balance_beta": float(args.generator_balance_beta),
                "paired_rf_sampling": bool(enable_paired_rf_sampling),
                "paired_rf_source_col": str(args.paired_rf_source_col),
                "paired_rf_group_source": str(paired_group_status),
                "paired_rf_group_count": int(paired_group_count),
                "gend_visual_adapt": bool(enable_gend_visual_adapt),
                "gend_adapt_strength": float(args.gend_adapt_strength),
                "gend_adapt_prob": float(args.gend_adapt_prob),
                "gend_visual_feature_count": int(len(visual_adapt_indices)),
                "scenario_focus": str(args.scenario_focus),
                "scenario_focus_weight": float(args.scenario_focus_weight),
                "hard_negative_file": (
                    None if args.hard_negative_file is None else str(args.hard_negative_file)
                ),
                "hard_negative_weight": float(args.hard_negative_weight),
                "hard_positive_file": (
                    None if args.hard_positive_file is None else str(args.hard_positive_file)
                ),
                "hard_positive_weight": float(args.hard_positive_weight),
            },
            "chosen_epoch_report": chosen_report,
        }
        with open(threshold_report_path, "w") as fp:
            json.dump(report_payload, fp, indent=2)
        print(f"✔ Threshold report saved to {threshold_report_path}")

    print("Learned alpha (AV causal weight):", model.alpha.item())
    print("Learned beta (Physical causal weight):", model.beta.item())
    if hasattr(model, "gamma"):
        try:
            print("Learned gamma (Lip-stream causal weight):", model.gamma.item())
        except Exception:
            pass
    if args.causal_weight > 0:
        # Report average causal penalty on the validation set
        model.eval()
        with torch.no_grad():
            lip_in = X_lip_val_t.to(device) if enable_lip_stream else None
            _, av_h, phys_h, lip_h, _ = model.branch_outputs(
                X_av_val.to(device),
                X_phys_val.to(device),
                lip_features=lip_in,
            )
            causal_pen = model.causal_penalty(av_h, phys_h, lip_h).item()
        print("Validation causal penalty:", causal_pen)
    print(f"✔ CFN model saved to {model_path}")
    if scaler is not None:
        print(f"✔ Scaler saved to {scaler_path}")


if __name__ == "__main__":
    main()
