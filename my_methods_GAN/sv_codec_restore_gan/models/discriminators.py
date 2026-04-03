from __future__ import annotations

import torch
from torch import nn


class _ConvDisc1D(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (16, 64, 128, 256)):
        super().__init__()
        layers = []
        in_ch = 1
        for ch in channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, ch, kernel_size=15, stride=2, padding=7),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_ch = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = wav.unsqueeze(1)
        feats = []
        for layer in self.body:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        score = self.head(x)
        return score, feats


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([_ConvDisc1D(), _ConvDisc1D(), _ConvDisc1D()])

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        outs = []
        x = wav
        for idx, disc in enumerate(self.discs):
            if idx > 0:
                x = nn.functional.avg_pool1d(x.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
            outs.append(disc(x))
        return outs


class MultiBandDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = _ConvDisc1D()
        self.mid = _ConvDisc1D()
        self.high = _ConvDisc1D()

    def _split_bands(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lightweight proxy bands by decimation and residuals.
        low = nn.functional.avg_pool1d(wav.unsqueeze(1), kernel_size=4, stride=1, padding=2).squeeze(1)
        mid = wav - low
        high = mid - nn.functional.avg_pool1d(mid.unsqueeze(1), kernel_size=2, stride=1, padding=1).squeeze(1)
        return low, mid, high

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        low, mid, high = self._split_bands(wav)
        return [self.low(low), self.mid(mid), self.high(high)]
