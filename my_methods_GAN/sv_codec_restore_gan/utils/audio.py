from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F


def _to_float32_tensor(wav_np: np.ndarray) -> torch.Tensor:
    if np.issubdtype(wav_np.dtype, np.integer):
        info = np.iinfo(wav_np.dtype)
        wav_np = wav_np.astype(np.float32) / max(abs(info.min), info.max)
    else:
        wav_np = wav_np.astype(np.float32)
    return torch.from_numpy(wav_np)


def _load_audio_with_soundfile(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # downmix to mono
    return torch.from_numpy(wav), int(sr)


def _load_audio_with_scipy(path: Path) -> tuple[torch.Tensor, int]:
    from scipy.io import wavfile

    sr, wav = wavfile.read(str(path))
    wav_np = np.asarray(wav)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    return _to_float32_tensor(wav_np), int(sr)


def load_audio_mono(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    path = Path(path).resolve()

    last_exc: Exception | None = None
    wav: torch.Tensor | None = None
    sr = sample_rate

    try:
        wav, sr = _load_audio_with_soundfile(path)
    except Exception as exc:
        last_exc = exc
        try:
            wav, sr = _load_audio_with_scipy(path)
        except Exception as exc2:
            last_exc = exc2

    if wav is None:
        raise RuntimeError(f"Failed to load audio file: {path}") from last_exc

    wav = wav.contiguous()
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
