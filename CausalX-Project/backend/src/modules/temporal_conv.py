import torch
import torch.nn as nn


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for idx, ch in enumerate(channels):
            dilation = 2 ** idx
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Conv1d(
                    current,
                    ch,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            current = ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        return out.mean(dim=-1)
