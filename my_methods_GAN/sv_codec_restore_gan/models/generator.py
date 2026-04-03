from __future__ import annotations

import torch
from torch import nn

from sv_codec_restore_gan.utils.audio import to_16k, to_48k


class GridNetBlock(nn.Module):
    def __init__(self, channels: int, hidden: int = 100, heads: int = 4):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2), groups=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=1),
        )
        self.time_gru = nn.GRU(input_size=channels, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.time_proj = nn.Linear(hidden * 2, channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        residual = x
        x = x + self.freq_conv(x)

        b, c, t, f = x.shape
        xt = x.mean(dim=-1).transpose(1, 2)  # [B, T, C]
        xt, _ = self.time_gru(xt)
        xt = self.time_proj(xt)
        xa, _ = self.attn(xt, xt, xt, need_weights=False)
        xt = self.norm(xt + xa)
        xt = xt.transpose(1, 2).unsqueeze(-1)  # [B, C, T, 1]
        x = x + xt
        return x + residual


class SVCodecRestoreGenerator(nn.Module):
    """CWS-TF-GridNet-style generator for codec restoration.

    Internal processing is done at 48 kHz and returns 16 kHz waveform.
    """

    def __init__(
        self,
        n_fft: int = 1536,
        hop_length: int = 768,
        win_length: int = 1536,
        emb_dim: int = 48,
        num_blocks: int = 5,
        hidden_units: int = 100,
        attn_heads: int = 4,
        cws_subbands: int = 3,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.cws_subbands = cws_subbands

        self.in_proj = nn.Conv2d(2, emb_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [GridNetBlock(emb_dim, hidden=hidden_units, heads=attn_heads) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv2d(emb_dim, 2, kernel_size=1)

    def _stft_ri(self, wav: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=wav.device)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        ri = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]
        return ri.permute(0, 1, 3, 2).contiguous()  # [B, 2, T, F]

    def _istft_ri(self, ri: torch.Tensor, length: int) -> torch.Tensor:
        # ri: [B,2,T,F]
        x = ri.permute(0, 1, 3, 2).contiguous()  # [B,2,F,T]
        spec = torch.complex(x[:, 0], x[:, 1])
        window = torch.hann_window(self.win_length, device=ri.device)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=length,
        )

    def _cws_split(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Split frequency bins into contiguous subbands.
        return list(torch.chunk(x, self.cws_subbands, dim=-1))

    def _cws_merge(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=-1)

    def forward(self, coded_16k: torch.Tensor) -> torch.Tensor:
        # 16k -> 48k internal processing
        coded_48k = to_48k(coded_16k)
        x = self._stft_ri(coded_48k)

        subbands = self._cws_split(x)
        out_subbands = []
        for s in subbands:
            h = self.in_proj(s)
            for blk in self.blocks:
                h = blk(h)
            out_subbands.append(self.out_proj(h))

        pred_ri = self._cws_merge(out_subbands)
        pred_res_48k = self._istft_ri(pred_ri, length=coded_48k.shape[-1])
        restored_48k = coded_48k + pred_res_48k
        restored_16k = to_16k(restored_48k)
        return restored_16k
