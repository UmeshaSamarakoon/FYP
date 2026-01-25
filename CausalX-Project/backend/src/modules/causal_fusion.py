import torch
import torch.nn as nn

class CausalFusionNetwork(nn.Module):
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

    def forward(self, av_features, physical_features):
        av_out = self.av_branch(av_features)
        phys_out = self.physical_branch(physical_features)

        fused = self.alpha * av_out + self.beta * phys_out
        return self.classifier(fused)
