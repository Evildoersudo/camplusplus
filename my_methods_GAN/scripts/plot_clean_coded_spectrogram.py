#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import wave

import matplotlib
import numpy as np
import torch
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot side-by-side spectrogram comparison for clean vs coded speech.",
    )
    parser.add_argument("--clean_wav", type=str, required=True, help="Path to clean wav/flac file.")
    parser.add_argument("--coded_wav", type=str, required=True, help="Path to coded wav/flac file (e.g., AMR-WB decoded wav).")
    parser.add_argument("--output_png", type=str, required=True, help="Output image path.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate for analysis.")
    parser.add_argument("--codec_name", type=str, default="AMR-WB", help="Codec name shown in subplot title.")

    parser.add_argument("--n_fft", type=int, default=1024, help="STFT FFT size.")
    parser.add_argument("--hop_length", type=int, default=256, help="STFT hop length.")
    parser.add_argument("--win_length", type=int, default=1024, help="STFT window length.")
    parser.add_argument("--max_freq", type=float, default=8000.0, help="Max frequency (Hz) shown on y-axis.")
    parser.add_argument("--dynamic_range_db", type=float, default=80.0, help="Color range in dB (vmin=vmax-dynamic_range_db).")

    parser.add_argument("--fig_width", type=float, default=18.0, help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=8.0, help="Figure height in inches.")
    parser.add_argument("--font_size", type=float, default=22.0, help="Base font size for all text in figure.")
    parser.add_argument("--title_size", type=float, default=26.0, help="Main title font size.")
    parser.add_argument("--subplot_title_size", type=float, default=23.0, help="Subplot title font size.")
    parser.add_argument("--label_size", type=float, default=21.0, help="Axis label font size.")
    parser.add_argument("--tick_size", type=float, default=18.0, help="Axis tick font size.")
    parser.add_argument("--colorbar_size", type=float, default=18.0, help="Colorbar label/tick font size.")
    parser.add_argument("--dpi", type=int, default=220, help="Output PNG dpi.")
    return parser.parse_args()


def _load_wav_with_wave(path: Path) -> tuple[torch.Tensor, int]:
    """Load PCM WAV without relying on external codecs."""

    with wave.open(str(path), "rb") as wf:
        if wf.getcomptype() != "NONE":
            raise RuntimeError(f"Compressed WAV is not supported by stdlib wave: {path}")

        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sample_width == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x_i32 = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign_mask = 1 << 23
        x_i32 = (x_i32 ^ sign_mask) - sign_mask
        x = x_i32.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")

    x = x.reshape(-1, num_channels).T
    return torch.from_numpy(x), int(sample_rate)


def load_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    load_errors: list[str] = []

    wav = None
    sr = None

    if path.suffix.lower() == ".wav":
        try:
            wav, sr = _load_wav_with_wave(path)
        except Exception as exc:
            load_errors.append(f"wave loader failed: {exc}")

    if wav is None:
        try:
            import soundfile as sf

            np_wav, np_sr = sf.read(str(path), dtype="float32", always_2d=True)
            wav = torch.from_numpy(np_wav.T)
            sr = int(np_sr)
        except Exception as exc:
            load_errors.append(f"soundfile loader failed: {exc}")

    if wav is None:
        try:
            wav, sr = torchaudio.load(str(path))
        except Exception as exc:
            load_errors.append(f"torchaudio loader failed: {exc}")
            raise RuntimeError(
                "Failed to load audio. Tried wave/soundfile/torchaudio loaders. "
                + " | ".join(load_errors)
            ) from exc

    if wav.ndim != 2:
        raise RuntimeError(f"Unexpected waveform shape for {path}: {tuple(wav.shape)}")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

    return wav.squeeze(0)


def compute_log_spectrogram_db(wav: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    window = torch.hann_window(win_length, device=wav.device)
    stft = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
    )
    magnitude = stft.abs().clamp_min(1e-10)
    spec_db = 20.0 * torch.log10(magnitude)
    return spec_db.cpu().numpy()


def setup_plot_style(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.subplot_title_size,
            "axes.labelsize": args.label_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
        }
    )


def plot_comparison(clean_db: np.ndarray, coded_db: np.ndarray, args: argparse.Namespace, out_path: Path) -> None:
    time_clean = clean_db.shape[1] * args.hop_length / float(args.sample_rate)
    time_coded = coded_db.shape[1] * args.hop_length / float(args.sample_rate)
    y_top = min(float(args.max_freq), args.sample_rate / 2.0)

    global_max = float(max(clean_db.max(), coded_db.max()))
    vmin = global_max - float(args.dynamic_range_db)
    vmax = global_max

    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height), sharey=True)

    im0 = axes[0].imshow(
        clean_db,
        origin="lower",
        aspect="auto",
        extent=[0.0, time_clean, 0.0, args.sample_rate / 2.0],
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
    )
    axes[0].set_title("Clean")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_ylim(0.0, y_top)

    im1 = axes[1].imshow(
        coded_db,
        origin="lower",
        aspect="auto",
        extent=[0.0, time_coded, 0.0, args.sample_rate / 2.0],
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
    )
    axes[1].set_title(f"Coded ({args.codec_name})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0.0, y_top)

    # Create a dedicated colorbar axis at the far right to avoid overlap with the second subplot.
    cax = fig.add_axes([0.92, 0.12, 0.018, 0.72])
    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label("Magnitude (dB)", fontsize=args.colorbar_size)
    cbar.ax.tick_params(labelsize=args.colorbar_size)

    fig.suptitle("Clean vs Coded Speech Spectrogram", fontsize=args.title_size)
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.1, top=0.86, wspace=0.12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_plot_style(args)

    clean_path = Path(args.clean_wav).resolve()
    coded_path = Path(args.coded_wav).resolve()
    out_path = Path(args.output_png).resolve()

    if not clean_path.exists():
        raise FileNotFoundError(f"clean_wav not found: {clean_path}")
    if not coded_path.exists():
        raise FileNotFoundError(f"coded_wav not found: {coded_path}")

    clean_wav = load_audio_mono(clean_path, args.sample_rate)
    coded_wav = load_audio_mono(coded_path, args.sample_rate)

    clean_db = compute_log_spectrogram_db(clean_wav, args.n_fft, args.hop_length, args.win_length)
    coded_db = compute_log_spectrogram_db(coded_wav, args.n_fft, args.hop_length, args.win_length)

    plot_comparison(clean_db, coded_db, args, out_path)

    print(f"Saved figure: {out_path}")
    print(f"clean: {clean_path}")
    print(f"coded: {coded_path}")


if __name__ == "__main__":
    main()
