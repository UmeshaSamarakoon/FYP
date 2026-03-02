import torch
import torch.nn as nn

class CausalFusionNetwork(nn.Module):
    """Minimal CFN with optional SCM-inspired consistency term.

    The causal consistency signal encourages the AV and physical branches to
    agree in their latent space. We keep it lightweight (MSE) so it can be
    toggled on/off in the trainer without changing inference outputs.
    """

    def __init__(self):
        super().__init__()

        # Audio–Visual causal branch
        self.av_branch = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        # Physical causal branch
        self.physical_branch = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        # Learnable causal fusion weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

        # Lightweight cross-modal attention to capture interactions
        self.fusion_attn = nn.MultiheadAttention(embed_dim=4, num_heads=1, batch_first=True)

    def branch_outputs(self, av_features, physical_features):
        """Return classifier output plus branch activations for auxiliary losses."""
        av_out = self.av_branch(av_features)
        phys_out = self.physical_branch(physical_features)

        # Attention over 2 tokens [av, phys]
        tokens = torch.stack([av_out, phys_out], dim=1)
        attn_out, _ = self.fusion_attn(tokens, tokens, tokens)
        attn_mean = attn_out.mean(dim=1)

        fused = self.alpha * av_out + self.beta * phys_out + 0.5 * attn_mean
        prob = self.classifier(fused)
        return prob, av_out, phys_out

    def causal_penalty(self, av_out, phys_out):
        # Mean squared error encourages latent alignment (SCM-style consistency)
        return torch.mean((av_out - phys_out) ** 2)

    def forward(self, av_features, physical_features):
        prob, _, _ = self.branch_outputs(av_features, physical_features)
        return prob


class CausalFusionNetworkV2(nn.Module):
    """Dim-flexible CFN with optional causal consistency penalty."""

    def __init__(self, av_dim: int, phys_dim: int):
        super().__init__()

        self.av_branch = nn.Sequential(
            nn.Linear(av_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.physical_branch = nn.Sequential(
            nn.Linear(phys_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 8)
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.classifier = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Cross-modal attention to fuse interactions
        self.fusion_attn = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)

    def branch_outputs(self, av_features, physical_features):
        av_out = self.av_branch(av_features)
        phys_out = self.physical_branch(physical_features)

        tokens = torch.stack([av_out, phys_out], dim=1)  # [B,2,8]
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
