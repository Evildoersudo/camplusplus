"""CA-AFC frontend model definitions.

This module implements the frontend proposed in `my_methods/note/method_way.md`.
The model keeps CAM++ frozen as the backend and learns a lightweight feature
compensation frontend that:

1. Encodes codec-degraded FBank features.
2. Encodes auxiliary temporal cues from codec speech.
3. Fuses the two branches with an attentive gate.
4. Predicts per-band attention and a residual compensation term.
5. Outputs an enhanced 80-dim FBank sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FrontendOutput:
    """Structured frontend outputs used by training and analysis scripts."""

    enhanced: torch.Tensor
    band_weights: torch.Tensor
    residual: torch.Tensor
    fused_hidden: torch.Tensor


class SpectralEncoder(nn.Module):
    """Encode codec FBank with shallow 2D convolutions while preserving time."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x.unsqueeze(1))
        return feat.mean(dim=-1).transpose(1, 2)


class AuxEncoder(nn.Module):
    """Encode pitch/voicing style auxiliary features with temporal Conv1D."""

    def __init__(self, aux_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(aux_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        return self.net(aux.transpose(1, 2)).transpose(1, 2)


class AttentiveFusion(nn.Module):
    """Fuse spectral and auxiliary branches with a learned gate."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.aux_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, spectral_hidden: torch.Tensor, aux_hidden: torch.Tensor) -> torch.Tensor:
        aux_proj = self.aux_proj(aux_hidden)
        alpha = torch.sigmoid(self.gate(torch.cat([spectral_hidden, aux_hidden], dim=-1)))
        return alpha * spectral_hidden + (1.0 - alpha) * aux_proj


class CAAFCFrontend(nn.Module):
    """Context-aware attentive feature compensation frontend.

    Input:
    - codec_fbank: [B, T, 80]
    - aux_feats: [B, T, d_aux]

    Output:
    - enhanced FBank with the same shape as codec_fbank.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        aux_dim: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim

        self.spectral_encoder = SpectralEncoder(hidden_dim=hidden_dim)
        self.aux_encoder = AuxEncoder(aux_dim=aux_dim, hidden_dim=hidden_dim)
        self.fusion = AttentiveFusion(hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.band_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, codec_fbank: torch.Tensor, aux_feats: torch.Tensor) -> FrontendOutput:
        spectral_hidden = self.spectral_encoder(codec_fbank)
        aux_hidden = self.aux_encoder(aux_feats)
        fused_hidden = self.dropout(self.fusion(spectral_hidden, aux_hidden))

        band_weights = self.band_attention(fused_hidden)
        residual = self.residual_head(fused_hidden)
        weighted = band_weights * codec_fbank
        enhanced = weighted + residual

        return FrontendOutput(
            enhanced=enhanced,
            band_weights=band_weights,
            residual=residual,
            fused_hidden=fused_hidden,
        )

