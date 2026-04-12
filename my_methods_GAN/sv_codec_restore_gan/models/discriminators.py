from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


_WINDOW_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _get_hann_window(n_fft: int, device: torch.device) -> torch.Tensor:
    key = (str(device), int(n_fft))
    window = _WINDOW_CACHE.get(key)
    if window is None:
        window = torch.hann_window(n_fft, device=device)
        _WINDOW_CACHE[key] = window
    return window


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


class _STFTDisc2D(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)

        layers = []
        in_ch = 1
        for ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, ch, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_ch = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1))

    def _stft_logmag(self, wav: torch.Tensor) -> torch.Tensor:
        window = _get_hann_window(self.win_length, wav.device)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        mag = spec.abs().clamp_min(1e-7)
        logmag = torch.log(mag)
        return logmag.unsqueeze(1)  # (B,1,F,T)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self._stft_logmag(wav)
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
        # Multi-resolution STFT discriminators for codec-artifact-sensitive supervision.
        self.discs = nn.ModuleList(
            [
                _STFTDisc2D(n_fft=256, hop_length=64, win_length=256),
                _STFTDisc2D(n_fft=512, hop_length=128, win_length=512),
                _STFTDisc2D(n_fft=1024, hop_length=256, win_length=1024),
            ]
        )

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        return [disc(wav) for disc in self.discs]


class MultiBandDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = _ConvDisc1D()
        self.mid = _ConvDisc1D()
        self.high = _ConvDisc1D()

    @staticmethod
    def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        cur = x.shape[-1]
        if cur > target_len:
            return x[..., :target_len]
        if cur < target_len:
            return F.pad(x, (0, target_len - cur))
        return x

    def _split_bands(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lightweight proxy bands by decimation and residuals.
        target_len = wav.shape[-1]
        low = nn.functional.avg_pool1d(wav.unsqueeze(1), kernel_size=4, stride=1, padding=2).squeeze(1)
        low = self._match_length(low, target_len)
        mid = wav - low
        high_smooth = nn.functional.avg_pool1d(mid.unsqueeze(1), kernel_size=2, stride=1, padding=1).squeeze(1)
        high_smooth = self._match_length(high_smooth, target_len)
        high = mid - high_smooth
        return low, mid, high

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        low, mid, high = self._split_bands(wav)
        return [self.low(low), self.mid(mid), self.high(high)]
