#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sv_codec_restore_gan.models.generator import SVCodecRestoreGenerator
from sv_codec_restore_gan.utils.audio import load_audio_mono


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run coded speech through a trained generator and plot clean/coded/restored spectrograms.",
    )
    p.add_argument("--clean_wav", type=str, required=True, help="Path to clean reference wav/flac.")
    p.add_argument("--coded_wav", type=str, required=True, help="Path to coded wav/flac (model input).")
    p.add_argument("--generator_ckpt", type=str, required=True, help="Checkpoint path for trained generator.")
    p.add_argument("--output_png", type=str, required=True, help="Output 3-panel spectrogram figure path.")
    p.add_argument("--output_restored_wav", type=str, default="", help="Optional output path for restored wav.")

    p.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate for load/inference.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    p.add_argument("--infer_chunk_seconds", type=float, default=8.0, help="Chunk window for generator inference, <=0 disables chunking.")
    p.add_argument("--infer_hop_seconds", type=float, default=8.0, help="Chunk hop for generator inference.")
    p.add_argument("--infer_chunk_batch_size", type=int, default=8, help="Batch size for chunked inference.")

    p.add_argument("--n_fft", type=int, default=1024, help="STFT FFT size for plotting.")
    p.add_argument("--hop_length", type=int, default=256, help="STFT hop length for plotting.")
    p.add_argument("--win_length", type=int, default=1024, help="STFT window length for plotting.")
    p.add_argument("--max_freq", type=float, default=8000.0, help="Maximum displayed frequency in Hz.")
    p.add_argument("--dynamic_range_db", type=float, default=80.0, help="Colormap dB dynamic range.")

    p.add_argument("--fig_width", type=float, default=27.0, help="Figure width in inches.")
    p.add_argument("--fig_height", type=float, default=8.0, help="Figure height in inches.")
    p.add_argument("--font_size", type=float, default=21.0, help="Global base font size.")
    p.add_argument("--title_size", type=float, default=30.0, help="Figure title font size.")
    p.add_argument("--subplot_title_size", type=float, default=26.0, help="Subplot title font size.")
    p.add_argument("--label_size", type=float, default=24.0, help="Axis label font size.")
    p.add_argument("--tick_size", type=float, default=19.0, help="Axis tick font size.")
    p.add_argument("--colorbar_size", type=float, default=19.0, help="Colorbar label and tick size.")
    p.add_argument("--dpi", type=int, default=220, help="Output image DPI.")

    p.add_argument("--plot_title", type=str, default="Clean vs Coded vs Restored Speech Spectrogram", help="Figure title text.")
    p.add_argument("--clean_title", type=str, default="Clean", help="Subplot title for clean waveform.")
    p.add_argument("--coded_title", type=str, default="Coded", help="Subplot title for coded waveform.")
    p.add_argument("--restored_title", type=str, default="Restored", help="Subplot title for restored waveform.")

    p.add_argument("--subplot_left", type=float, default=0.05, help="Left margin for subplot area.")
    p.add_argument("--subplot_right", type=float, default=0.90, help="Right margin for subplot area.")
    p.add_argument("--subplot_bottom", type=float, default=0.12, help="Bottom margin for subplot area.")
    p.add_argument("--subplot_top", type=float, default=0.86, help="Top margin for subplot area.")
    p.add_argument("--subplot_wspace", type=float, default=0.12, help="Horizontal spacing between subplots.")
    p.add_argument("--colorbar_left", type=float, default=0.92, help="Colorbar left position in figure fraction.")
    p.add_argument("--colorbar_bottom", type=float, default=0.13, help="Colorbar bottom position in figure fraction.")
    p.add_argument("--colorbar_width", type=float, default=0.018, help="Colorbar width in figure fraction.")
    p.add_argument("--colorbar_height", type=float, default=0.72, help="Colorbar height in figure fraction.")
    return p.parse_args()


def build_generator_from_ckpt(ckpt_path: Path, device: torch.device) -> SVCodecRestoreGenerator:
    state = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}

    model = SVCodecRestoreGenerator(
        emb_dim=int(ckpt_args.get("emb_dim", 64)),
        num_blocks=int(ckpt_args.get("num_blocks", 6)),
        hidden_units=int(ckpt_args.get("hidden_units", 128)),
        attn_heads=int(ckpt_args.get("attn_heads", 4)),
        cws_subbands=int(ckpt_args.get("cws_subbands", 3)),
        n_fft=int(ckpt_args.get("n_fft", 512)),
        hop_length=int(ckpt_args.get("hop_length", 128)),
        win_length=int(ckpt_args.get("win_length", 512)),
    )

    gen_state = state["generator"] if isinstance(state, dict) and "generator" in state else state
    model.load_state_dict(gen_state, strict=True)
    model.to(device).eval()
    return model


def run_generator_chunked(
    generator: SVCodecRestoreGenerator,
    coded: torch.Tensor,
    device: torch.device,
    chunk_samples: int,
    hop_samples: int,
    chunk_batch_size: int,
) -> torch.Tensor:
    if chunk_samples <= 0 or hop_samples <= 0 or coded.numel() <= chunk_samples:
        return generator(coded.unsqueeze(0).to(device)).squeeze(0).detach().cpu()

    out = torch.zeros_like(coded)
    wsum = torch.zeros_like(coded)
    win = torch.hann_window(chunk_samples, device="cpu")
    pad = max(0, chunk_samples - hop_samples)
    coded_pad = F.pad(coded, (pad, pad)) if pad > 0 else coded

    total = coded_pad.numel()
    starts = list(range(0, max(1, total - chunk_samples + 1), hop_samples))
    if not starts:
        starts = [0]

    chunk_batch_size = max(1, int(chunk_batch_size))
    for bi in range(0, len(starts), chunk_batch_size):
        batch_starts = starts[bi : bi + chunk_batch_size]
        chunks = [coded_pad[s : s + chunk_samples] for s in batch_starts]
        batch = torch.stack(chunks, dim=0).to(device)
        preds = generator(batch).detach().cpu()

        for j, s in enumerate(batch_starts):
            pred = preds[j] * win
            os = s - pad
            oe = os + chunk_samples
            cs = 0
            ce = chunk_samples

            if os < 0:
                cs = -os
                os = 0
            if oe > coded.numel():
                ce -= oe - coded.numel()
                oe = coded.numel()
            if os < oe and cs < ce:
                out[os:oe] += pred[cs:ce]
                wsum[os:oe] += win[cs:ce]

    mask = wsum <= 1e-8
    out = out / wsum.clamp_min(1e-8)
    if mask.any():
        out[mask] = coded[mask]
    return out


def save_wav_pcm16(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    x = wav.detach().cpu().flatten().clamp(-1.0, 1.0)
    pcm = (x * 32767.0).round().to(torch.int16).numpy().tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)


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
    return (20.0 * torch.log10(magnitude)).cpu().numpy()


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


def plot_three_way(clean_db: np.ndarray, coded_db: np.ndarray, restored_db: np.ndarray, args: argparse.Namespace, out_path: Path) -> None:
    y_top = min(float(args.max_freq), args.sample_rate / 2.0)

    vmax = float(max(clean_db.max(), coded_db.max(), restored_db.max()))
    vmin = vmax - float(args.dynamic_range_db)

    fig, axes = plt.subplots(2, 2, figsize=(args.fig_width, args.fig_height), sharey=True)

    # 2x2 placement: clean(top-left), restored(top-right), coded(bottom-left), bottom-right left blank.
    panel_defs = [
        (axes[0, 0], clean_db, args.clean_title),
        (axes[0, 1], restored_db, args.restored_title),
        (axes[1, 0], coded_db, args.coded_title),
    ]

    im = None
    for i, (ax, mat, title) in enumerate(panel_defs):
        duration = mat.shape[1] * args.hop_length / float(args.sample_rate)
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[0.0, duration, 0.0, args.sample_rate / 2.0],
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0.0, y_top)
        if i in (0, 2):
            ax.set_ylabel("Frequency (Hz)")

    axes[1, 1].axis("off")

    fig.suptitle(args.plot_title, fontsize=args.title_size)
    fig.subplots_adjust(
        left=args.subplot_left,
        right=args.subplot_right,
        bottom=args.subplot_bottom,
        top=args.subplot_top,
        wspace=args.subplot_wspace,
    )

    cax = fig.add_axes([args.colorbar_left, args.colorbar_bottom, args.colorbar_width, args.colorbar_height])
    assert im is not None
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Magnitude (dB)", fontsize=args.colorbar_size)
    cbar.ax.tick_params(labelsize=args.colorbar_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_plot_style(args)

    clean_path = Path(args.clean_wav).resolve()
    coded_path = Path(args.coded_wav).resolve()
    ckpt_path = Path(args.generator_ckpt).resolve()
    out_png = Path(args.output_png).resolve()
    out_restored = Path(args.output_restored_wav).resolve() if args.output_restored_wav else None

    if not clean_path.exists():
        raise FileNotFoundError(f"clean_wav not found: {clean_path}")
    if not coded_path.exists():
        raise FileNotFoundError(f"coded_wav not found: {coded_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"generator_ckpt not found: {ckpt_path}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    clean = load_audio_mono(clean_path, sample_rate=args.sample_rate)
    coded = load_audio_mono(coded_path, sample_rate=args.sample_rate)

    generator = build_generator_from_ckpt(ckpt_path, device)

    chunk_samples = int(max(0.0, float(args.infer_chunk_seconds)) * float(args.sample_rate))
    hop_samples = int(max(0.0, float(args.infer_hop_seconds)) * float(args.sample_rate))

    with torch.no_grad():
        restored = run_generator_chunked(
            generator,
            coded.contiguous(),
            device,
            chunk_samples=chunk_samples,
            hop_samples=hop_samples,
            chunk_batch_size=max(1, int(args.infer_chunk_batch_size)),
        )

    if out_restored is not None:
        save_wav_pcm16(out_restored, restored, sample_rate=args.sample_rate)

    clean_db = compute_log_spectrogram_db(clean, args.n_fft, args.hop_length, args.win_length)
    coded_db = compute_log_spectrogram_db(coded, args.n_fft, args.hop_length, args.win_length)
    restored_db = compute_log_spectrogram_db(restored, args.n_fft, args.hop_length, args.win_length)

    plot_three_way(clean_db, coded_db, restored_db, args, out_png)

    print(f"Saved figure: {out_png}")
    if out_restored is not None:
        print(f"Saved restored wav: {out_restored}")
    print(f"clean: {clean_path}")
    print(f"coded: {coded_path}")
    print(f"generator_ckpt: {ckpt_path}")
    print(f"device: {device}")


if __name__ == "__main__":
    main()
