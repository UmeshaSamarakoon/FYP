import warnings

import torch
import torch.nn as nn


class CausalFusionNetworkV2(nn.Module):
    """Supported CFN architecture: AV branch + physical branch + attention fusion."""

    def __init__(
        self,
        av_dim: int,
        phys_dim: int,
        lip_dim: int = 0,
        enable_causal_breach_head: bool = False,
        enable_av_input_layernorm: bool = False,
        **kwargs,
    ):
        super().__init__()
        if int(lip_dim or 0) > 0:
            raise ValueError(
                "Lip-branch CFN checkpoints are no longer supported. "
                "Use the active two-branch V2 checkpoints instead."
            )

        _ = (enable_causal_breach_head, kwargs)
        self.enable_av_input_layernorm = bool(enable_av_input_layernorm)
        self.av_input_ln = nn.LayerNorm(int(av_dim)) if self.enable_av_input_layernorm else None

        self.av_branch = nn.Sequential(
            nn.Linear(av_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        self.physical_branch = nn.Sequential(
            nn.Linear(phys_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.classifier = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.fusion_attn = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)

    def branch_outputs(self, av_features, physical_features):
        if self.av_input_ln is not None:
            av_features = self.av_input_ln(av_features)
        av_out = self.av_branch(av_features)
        phys_out = self.physical_branch(physical_features)

        tokens = torch.stack([av_out, phys_out], dim=1)
        attn_out, _ = self.fusion_attn(tokens, tokens, tokens)
        attn_mean = attn_out.mean(dim=1)

        fused = self.alpha * av_out + self.beta * phys_out + 0.5 * attn_mean
        prob = self.classifier(fused)
        return prob, av_out, phys_out

    def causal_penalty(self, av_out, phys_out):
        return torch.mean((av_out - phys_out) ** 2)

    def forward(self, av_features, physical_features):
        prob, _, _ = self.branch_outputs(av_features, physical_features)
        return prob


class CausalFusionNetwork(CausalFusionNetworkV2):
    """
    Backward-compatible alias that preserves old imports while using the
    single supported CFN implementation underneath.
    """

    def __init__(
        self,
        enable_causal_breach_head: bool = False,
        enable_av_input_layernorm: bool = False,
        **kwargs,
    ):
        warnings.warn(
            "CausalFusionNetwork is deprecated. Use CausalFusionNetworkV2 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            av_dim=3,
            phys_dim=2,
            lip_dim=0,
            enable_causal_breach_head=enable_causal_breach_head,
            enable_av_input_layernorm=enable_av_input_layernorm,
            **kwargs,
        )
