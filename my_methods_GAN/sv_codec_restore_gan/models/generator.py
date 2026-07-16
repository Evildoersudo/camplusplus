from __future__ import annotations

import torch
from torch import nn

from .init_utils import apply_model_init


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
        self.freq_gru = nn.GRU(input_size=channels, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.freq_proj = nn.Linear(hidden * 2, channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)
        self.norm_t = nn.LayerNorm(channels)
        self.norm_f = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        residual = x
        x = x + self.freq_conv(x)

        b, c, t, f = x.shape
        xt = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)  # [B*F, T, C]
        xt, _ = self.time_gru(xt)
        xt = self.time_proj(xt)
        xa, _ = self.attn(xt, xt, xt, need_weights=False)
        xt = self.norm_t(xt + xa)
        xt = xt.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()
        x = x + xt

        xf = x.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)  # [B*T, F, C]
        xf, _ = self.freq_gru(xf)
        xf = self.freq_proj(xf)
        xf = self.norm_f(xf)
        xf = xf.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()
        x = x + xf
        return x


class SVCodecRestoreGenerator(nn.Module):
    """CWS-TF-GridNet-style generator for codec restoration.

    Internal processing is done directly at 16 kHz.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        emb_dim: int = 48,
        num_blocks: int = 5,
        hidden_units: int = 100,
        attn_heads: int = 4,
        cws_subbands: int = 3,
        cws_mode: str = "uniform",
        cws_band_edges: tuple[int, ...] | list[int] | str | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.cws_subbands = cws_subbands
        self.cws_mode = str(cws_mode).lower()
        if self.cws_mode not in {"uniform", "nonuniform"}:
            raise ValueError(f"Unsupported cws_mode={cws_mode!r}; expected 'uniform' or 'nonuniform'.")
        self.cws_band_edges = self._normalize_cws_band_edges(cws_band_edges)
        if self.cws_mode == "nonuniform" and len(self.cws_band_edges) != self.cws_subbands + 1:
            raise ValueError(
                f"cws_band_edges length must be cws_subbands+1; "
                f"got len={len(self.cws_band_edges)} cws_subbands={self.cws_subbands}."
            )

        self.in_proj = nn.Conv2d(2 * cws_subbands, emb_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [GridNetBlock(emb_dim, hidden=hidden_units, heads=attn_heads) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv2d(emb_dim, 2 * cws_subbands, kernel_size=1)

        # Cache STFT window to avoid per-step allocations.
        self.register_buffer("_stft_window", torch.hann_window(self.win_length), persistent=False)
        #apply_model_init(self, nonlinearity="relu")

    def _normalize_cws_band_edges(self, edges: tuple[int, ...] | list[int] | str | None) -> tuple[int, ...]:
        if edges is None or edges == "":
            if self.cws_mode == "nonuniform":
                return (0, 64, 160, self.n_fft // 2 + 1)
            return ()
        if isinstance(edges, str):
            edges = tuple(int(x.strip()) for x in edges.split(",") if x.strip())
        else:
            edges = tuple(int(x) for x in edges)
        if len(edges) < 2:
            raise ValueError(f"cws_band_edges must contain at least 2 values, got {edges}.")
        if edges[0] != 0:
            raise ValueError(f"cws_band_edges must start with 0, got {edges}.")
        if any(e <= s for s, e in zip(edges[:-1], edges[1:])):
            raise ValueError(f"cws_band_edges must be strictly increasing, got {edges}.")
        return edges

    def _stft_ri(self, wav: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._stft_window,
            return_complex=True,
        )
        ri = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]
        return ri.permute(0, 1, 3, 2).contiguous()  # [B, 2, T, F]

    def _istft_ri(self, ri: torch.Tensor, length: int) -> torch.Tensor:
        # ri: [B,2,T,F]
        x = ri.permute(0, 1, 3, 2).contiguous()  # [B,2,F,T]
        spec = torch.complex(x[:, 0], x[:, 1])
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._stft_window,
            length=length,
        )

    def _pad_freq(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        freq = x.shape[-1]
        rem = freq % self.cws_subbands
        if rem == 0:
            return x, freq
        pad = self.cws_subbands - rem
        return nn.functional.pad(x, (0, pad)), freq

    def _cws_concat(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if self.cws_mode == "nonuniform":
            orig_freq = x.shape[-1]
            edges = self.cws_band_edges
            if edges[-1] != orig_freq:
                raise ValueError(
                    f"nonuniform cws_band_edges last value must equal current STFT freq bins. "
                    f"got edges[-1]={edges[-1]}, freq={orig_freq}."
                )
            chunks = []
            lengths = []
            for start, end in zip(edges[:-1], edges[1:]):
                band = x[..., start:end]
                chunks.append(band)
                lengths.append(end - start)
            max_len = max(lengths)
            padded = []
            for band in chunks:
                pad_len = max_len - band.shape[-1]
                if pad_len > 0:
                    band = nn.functional.pad(band, (0, pad_len))
                padded.append(band)
            x_cat = torch.cat(padded, dim=1)
            return x_cat, orig_freq

        x_pad, orig_freq = self._pad_freq(x)
        chunks = torch.chunk(x_pad, self.cws_subbands, dim=-1)
        x_cat = torch.cat(chunks, dim=1)
        return x_cat, orig_freq

    def _cws_merge(self, x: torch.Tensor, orig_freq: int) -> torch.Tensor:
        if self.cws_mode == "nonuniform":
            chunks = torch.chunk(x, self.cws_subbands, dim=1)
            bands = []
            for chunk, start, end in zip(chunks, self.cws_band_edges[:-1], self.cws_band_edges[1:]):
                bands.append(chunk[..., : end - start])
            merged = torch.cat(bands, dim=-1)
            return merged[..., :orig_freq]

        chunks = torch.chunk(x, self.cws_subbands, dim=1)
        merged = torch.cat(chunks, dim=-1)
        return merged[..., :orig_freq]

    def forward(self, coded_16k: torch.Tensor):
        x = self._stft_ri(coded_16k)
        x_cws, orig_freq = self._cws_concat(x)

        h = self.in_proj(x_cws)
        for blk in self.blocks:
            h = blk(h)
        pred_ri_cws = self.out_proj(h)

        pred_ri = self._cws_merge(pred_ri_cws, orig_freq)
        pred_res = self._istft_ri(pred_ri, length=coded_16k.shape[-1])
        restored = coded_16k + pred_res
        return restored
