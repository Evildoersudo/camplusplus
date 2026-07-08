"""Classic frontend feature helpers used by CA-AFC tools."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F


def _as_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() != 2:
        raise ValueError(f"Expected waveform shape [channels, samples], got {tuple(wav.shape)}")
    if wav.shape[0] == 1:
        return wav.squeeze(0)
    return wav.mean(dim=0)


def load_audio_mono(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    """Load one audio file as mono and resample if needed."""

    path = Path(path).resolve()
    try:
        wav_np, sr = sf.read(str(path), always_2d=False)
        wav = torch.as_tensor(wav_np, dtype=torch.float32)
        if wav.dim() == 2:
            wav = wav.mean(dim=1)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                str(tmp.name),
            ]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg decode failed for {path}: {result.stderr.strip()}") from None
            wav_np, sr = sf.read(tmp.name, always_2d=False)
            wav = torch.as_tensor(wav_np, dtype=torch.float32)
            if wav.dim() == 2:
                wav = wav.mean(dim=1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, sample_rate).squeeze(0)
    return wav.contiguous()


def _kaldi_waveform(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() != 1:
        raise ValueError(f"Expected mono waveform shape [samples], got {tuple(wav.shape)}")
    return wav.unsqueeze(0)


def compute_fbank_feature(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> torch.Tensor:
    """Compute Kaldi-style FBank features with shape [T, F]."""

    feat = torchaudio.compliance.kaldi.fbank(
        _kaldi_waveform(wav),
        sample_frequency=sample_rate,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        use_energy=False,
        dither=0.0,
    )
    return feat.contiguous()


def compute_mfcc_feature(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    num_ceps: int = 13,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> torch.Tensor:
    """Compute Kaldi-style MFCC features with shape [T, C]."""

    feat = torchaudio.compliance.kaldi.mfcc(
        _kaldi_waveform(wav),
        sample_frequency=sample_rate,
        num_mel_bins=num_mel_bins,
        num_ceps=num_ceps,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        use_energy=False,
        dither=0.0,
    )
    return feat.contiguous()


def apply_cmvn(feat: torch.Tensor, variance_norm: bool = False, eps: float = 1e-5) -> torch.Tensor:
    """Apply utterance-level CMVN to a [T, D] feature matrix."""

    if feat.dim() != 2:
        raise ValueError(f"Expected feature shape [T, D], got {tuple(feat.shape)}")
    mean = feat.mean(dim=0, keepdim=True)
    normed = feat - mean
    if variance_norm:
        std = normed.pow(2).mean(dim=0, keepdim=True).add(eps).sqrt()
        normed = normed / std
    return normed.contiguous()


def _interp_1d(values: torch.Tensor, target_len: int) -> torch.Tensor:
    if values.numel() == target_len:
        return values
    if values.numel() <= 1:
        return values.new_full((target_len,), float(values.reshape(-1)[0]) if values.numel() else 0.0)
    resized = F.interpolate(values.view(1, 1, -1), size=target_len, mode="linear", align_corners=False)
    return resized.view(-1)


def compute_pitch_proxy(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    frame_count: int | None = None,
    frame_time: float = 0.01,
) -> torch.Tensor:
    """Compute a lightweight pitch proxy aligned to frame_count."""

    try:
        pitch = torchaudio.functional.detect_pitch_frequency(
            wav.unsqueeze(0),
            sample_rate=sample_rate,
            frame_time=frame_time,
        ).squeeze(0)
    except Exception:
        pitch = wav.new_zeros((1,))

    if frame_count is None:
        return pitch.contiguous()
    return _interp_1d(pitch, frame_count).contiguous()
