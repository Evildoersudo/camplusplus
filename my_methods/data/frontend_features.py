"""Reusable frontend feature extraction utilities.

This module focuses on classic frontend processing blocks that are easy to
reuse in experiments and visualization:
- waveform loading
- FBank extraction
- MFCC extraction
- CMVN
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi


def load_audio_mono(path: Path, sample_rate: int) -> torch.Tensor:
    """Load audio as mono waveform and resample to the requested sample rate."""

    wav, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[0] > 1:
        wav = wav[:1]
    return wav.squeeze(0)


def compute_fbank_feature(
    wav: torch.Tensor,
    sample_rate: int,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    dither: float = 0.0,
) -> torch.Tensor:
    """Compute log-Mel FBank feature with Kaldi-compatible settings."""

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    feat = Kaldi.fbank(
        wav,
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_rate,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        dither=dither,
    )
    return feat


def compute_mfcc_feature(
    wav: torch.Tensor,
    sample_rate: int,
    num_mel_bins: int = 80,
    num_ceps: int = 13,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    dither: float = 0.0,
) -> torch.Tensor:
    """Compute MFCC feature with Kaldi-compatible settings."""

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    feat = Kaldi.mfcc(
        wav,
        num_mel_bins=num_mel_bins,
        num_ceps=num_ceps,
        sample_frequency=sample_rate,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        dither=dither,
    )
    return feat


def apply_cmvn(feature: torch.Tensor, variance_norm: bool = True, eps: float = 1e-5) -> torch.Tensor:
    """Apply utterance-level CMVN to a time-major feature matrix."""

    mean = feature.mean(dim=0, keepdim=True)
    centered = feature - mean
    if not variance_norm:
        return centered
    std = centered.std(dim=0, keepdim=True).clamp_min(eps)
    return centered / std

