import torch
import torch.nn as nn


class CausalFusionNetwork(nn.Module):
    """Minimal CFN with optional SCM-inspired consistency term."""

    def __init__(
        self,
        enable_multitask: bool = False,
        enable_causal_breach_head: bool = False,
        enable_av_input_layernorm: bool = False,
    ):
        super().__init__()
        self.enable_multitask = bool(enable_multitask)
        self.enable_causal_breach_head = bool(enable_causal_breach_head)
        self.enable_av_input_layernorm = bool(enable_av_input_layernorm)
        self.uses_lip_stream = False
        self.av_input_ln = nn.LayerNorm(3) if self.enable_av_input_layernorm else None

        self.av_branch = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.physical_branch = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.classifier = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )
        if self.enable_multitask:
            self.video_classifier = nn.Sequential(
                nn.Linear(4, 1),
                nn.Sigmoid(),
            )
            self.audio_classifier = nn.Sequential(
                nn.Linear(4, 1),
                nn.Sigmoid(),
            )
        else:
            self.video_classifier = None
            self.audio_classifier = None
        if self.enable_causal_breach_head:
            self.causal_breach_head = nn.Sequential(
                nn.Linear(4, 1),
                nn.Sigmoid(),
            )
        else:
            self.causal_breach_head = None

        self.fusion_attn = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)

    def _fused_latent(self, av_features, physical_features, lip_features=None):  # noqa: ARG002
        if self.av_input_ln is not None:
            av_features = self.av_input_ln(av_features)
        av_out = self.av_branch(av_features)
        phys_out = self.physical_branch(physical_features)
        tokens = torch.stack([av_out, phys_out], dim=1)
        attn_out, _ = self.fusion_attn(tokens, tokens, tokens)
        attn_mean = attn_out.mean(dim=1)
        fused = self.alpha * av_out + self.beta * phys_out + 0.5 * attn_mean
        return fused, av_out, phys_out, None

    def multitask_outputs(self, av_features, physical_features, lip_features=None):
        fused, av_out, phys_out, lip_out = self._fused_latent(
            av_features,
            physical_features,
            lip_features=lip_features,
        )
        prob = self.classifier(fused)
        video_prob = self.video_classifier(av_out) if self.enable_multitask else None
        audio_prob = self.audio_classifier(phys_out) if self.enable_multitask else None
        causal_breach_prob = (
            self.causal_breach_head(fused) if self.enable_causal_breach_head else None
        )
        return prob, av_out, phys_out, lip_out, video_prob, audio_prob, causal_breach_prob

    def branch_outputs(self, av_features, physical_features, lip_features=None):
        prob, av_out, phys_out, lip_out, _, _, causal_breach_prob = self.multitask_outputs(
            av_features,
            physical_features,
            lip_features=lip_features,
        )
        return prob, av_out, phys_out, lip_out, causal_breach_prob

    def causal_penalty(self, av_out, phys_out, lip_out=None):
        penalty = torch.mean((av_out - phys_out) ** 2)
        if lip_out is not None:
            penalty = penalty + 0.5 * (
                torch.mean((av_out - lip_out) ** 2) + torch.mean((phys_out - lip_out) ** 2)
            )
        return penalty

    def forward(self, av_features, physical_features, lip_features=None):
        prob, _, _, _, _ = self.branch_outputs(
            av_features,
            physical_features,
            lip_features=lip_features,
        )
        return prob


class CausalFusionNetworkV2(nn.Module):
    """Dim-flexible CFN with optional lip stream and causal-breach head."""

    def __init__(
        self,
        av_dim: int,
        phys_dim: int,
        enable_multitask: bool = False,
        lip_dim: int = 0,
        enable_causal_breach_head: bool = False,
        enable_av_input_layernorm: bool = False,
    ):
        super().__init__()
        self.enable_multitask = bool(enable_multitask)
        self.enable_causal_breach_head = bool(enable_causal_breach_head)
        self.enable_av_input_layernorm = bool(enable_av_input_layernorm)
        self.uses_lip_stream = int(lip_dim) > 0
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
        if self.uses_lip_stream:
            self.lip_branch = nn.Sequential(
                nn.Linear(int(lip_dim), 16),
                nn.ReLU(),
                nn.Linear(16, 8),
            )
        else:
            self.lip_branch = None

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        if self.uses_lip_stream:
            self.gamma = nn.Parameter(torch.tensor(0.5))

        self.classifier = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        if self.enable_multitask:
            self.video_classifier = nn.Sequential(
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )
            self.audio_classifier = nn.Sequential(
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )
        else:
            self.video_classifier = None
            self.audio_classifier = None
        if self.enable_causal_breach_head:
            self.causal_breach_head = nn.Sequential(
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )
        else:
            self.causal_breach_head = None

        self.fusion_attn = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)

    def _fused_latent(self, av_features, physical_features, lip_features=None):
        if self.av_input_ln is not None:
            av_features = self.av_input_ln(av_features)
        av_out = self.av_branch(av_features)
        phys_out = self.physical_branch(physical_features)

        lip_out = None
        if self.lip_branch is not None and lip_features is not None:
            lip_out = self.lip_branch(lip_features)

        tokens = [av_out, phys_out]
        if lip_out is not None:
            tokens.append(lip_out)
        stacked = torch.stack(tokens, dim=1)
        attn_out, _ = self.fusion_attn(stacked, stacked, stacked)
        attn_mean = attn_out.mean(dim=1)

        fused = self.alpha * av_out + self.beta * phys_out + 0.5 * attn_mean
        if lip_out is not None:
            fused = fused + self.gamma * lip_out
        return fused, av_out, phys_out, lip_out

    def multitask_outputs(self, av_features, physical_features, lip_features=None):
        fused, av_out, phys_out, lip_out = self._fused_latent(
            av_features,
            physical_features,
            lip_features=lip_features,
        )
        prob = self.classifier(fused)
        video_prob = self.video_classifier(av_out) if self.enable_multitask else None
        audio_prob = self.audio_classifier(phys_out) if self.enable_multitask else None
        causal_breach_prob = (
            self.causal_breach_head(fused) if self.enable_causal_breach_head else None
        )
        return prob, av_out, phys_out, lip_out, video_prob, audio_prob, causal_breach_prob

    def branch_outputs(self, av_features, physical_features, lip_features=None):
        prob, av_out, phys_out, lip_out, _, _, causal_breach_prob = self.multitask_outputs(
            av_features,
            physical_features,
            lip_features=lip_features,
        )
        return prob, av_out, phys_out, lip_out, causal_breach_prob

    def causal_penalty(self, av_out, phys_out, lip_out=None):
        penalty = torch.mean((av_out - phys_out) ** 2)
        if lip_out is not None:
            penalty = penalty + 0.5 * (
                torch.mean((av_out - lip_out) ** 2) + torch.mean((phys_out - lip_out) ** 2)
            )
        return penalty

    def forward(self, av_features, physical_features, lip_features=None):
        prob, _, _, _, _ = self.branch_outputs(
            av_features,
            physical_features,
            lip_features=lip_features,
        )
        return prob
