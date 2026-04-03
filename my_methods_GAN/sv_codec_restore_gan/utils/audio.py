from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F


def load_audio_mono(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(Path(path).resolve()))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, sample_rate).squeeze(0)
    return wav.contiguous()


def crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    if wav.numel() >= target_len:
        start = torch.randint(0, wav.numel() - target_len + 1, (1,)).item()
        return wav[start : start + target_len].contiguous()
    return F.pad(wav, (0, target_len - wav.numel())).contiguous()


def to_48k(wav16k: torch.Tensor) -> torch.Tensor:
    return torchaudio.functional.resample(wav16k.unsqueeze(0), 16000, 48000).squeeze(0)


def to_16k(wav48k: torch.Tensor) -> torch.Tensor:
    return torchaudio.functional.resample(wav48k.unsqueeze(0), 48000, 16000).squeeze(0)
