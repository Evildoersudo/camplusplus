"""Extract classic frontend features from one audio file and generate plots.

Outputs include:
- waveform plot
- FBank heatmap
- MFCC heatmap
- CMVN(FBank) heatmap
- CMVN(MFCC) heatmap
- `.npy` feature dumps
- summary JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_methods.data.frontend_features import (
    apply_cmvn,
    compute_fbank_feature,
    compute_mfcc_feature,
    load_audio_mono,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Read one audio file, compute FBank/MFCC/CMVN, and save plots.")
    parser.add_argument("--input_wav", type=str, default="", help="Input wav/flac audio path")
    parser.add_argument("--wav_scp", type=str, default="", help="Optional wav.scp used to resolve utt_id to the real audio path")
    parser.add_argument("--utt_id", type=str, default="", help="Optional utterance id looked up from wav.scp")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory used to save plots, numpy files, and summary JSON")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target audio sample rate")
    parser.add_argument("--num_mel_bins", type=int, default=80, help="FBank mel bin count and MFCC mel filterbank count")
    parser.add_argument("--num_ceps", type=int, default=13, help="MFCC cepstral coefficient count")
    parser.add_argument("--frame_length_ms", type=float, default=25.0, help="Frame length in milliseconds")
    parser.add_argument("--frame_shift_ms", type=float, default=10.0, help="Frame shift in milliseconds")
    parser.add_argument("--variance_norm", action="store_true", help="Apply full CMVN (mean + variance). Default is mean normalization only.")
    return parser.parse_args()


def load_wav_scp(path: Path):
    """Load a Kaldi wav.scp file into a `utt -> audio_path` mapping."""

    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            utt, audio_path = line.split(maxsplit=1)
            mapping[utt] = audio_path
    return mapping


def resolve_input_audio(args) -> Path:
    """Resolve the input audio either directly or through wav.scp + utt_id."""

    if args.input_wav:
        path = Path(args.input_wav).resolve()
        if path.exists():
            return path

    if args.wav_scp and args.utt_id:
        wav_scp = Path(args.wav_scp).resolve()
        mapping = load_wav_scp(wav_scp)
        if args.utt_id not in mapping:
            raise KeyError(f"utt_id not found in wav.scp: {args.utt_id}")
        path = Path(mapping[args.utt_id]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Resolved path from wav.scp does not exist: {path}")
        return path

    if args.input_wav:
        raise FileNotFoundError(
            f"input_wav does not exist: {Path(args.input_wav).resolve()}. "
            "If you are using CN-Celeb recipe data, prefer --wav_scp + --utt_id."
        )

    raise ValueError("You must provide either --input_wav or (--wav_scp and --utt_id).")


def save_feature_plot(feature: np.ndarray, title: str, out_path: Path):
    """Save one feature matrix as a heatmap image."""

    plt.figure(figsize=(10, 4))
    plt.imshow(feature.T, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Dimension")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_waveform_plot(wav: np.ndarray, sample_rate: int, out_path: Path):
    """Save waveform plot for the input audio."""

    time_axis = np.arange(wav.shape[0]) / float(sample_rate)
    plt.figure(figsize=(10, 3))
    plt.plot(time_axis, wav, linewidth=0.8)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    input_wav = resolve_input_audio(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wav = load_audio_mono(input_wav, sample_rate=args.sample_rate)
    fbank = compute_fbank_feature(
        wav,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
        frame_length_ms=args.frame_length_ms,
        frame_shift_ms=args.frame_shift_ms,
    )
    mfcc = compute_mfcc_feature(
        wav,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
        num_ceps=args.num_ceps,
        frame_length_ms=args.frame_length_ms,
        frame_shift_ms=args.frame_shift_ms,
    )
    fbank_cmvn = apply_cmvn(fbank, variance_norm=args.variance_norm)
    mfcc_cmvn = apply_cmvn(mfcc, variance_norm=args.variance_norm)

    np.save(output_dir / "fbank.npy", fbank.numpy())
    np.save(output_dir / "mfcc.npy", mfcc.numpy())
    np.save(output_dir / "fbank_cmvn.npy", fbank_cmvn.numpy())
    np.save(output_dir / "mfcc_cmvn.npy", mfcc_cmvn.numpy())

    save_waveform_plot(wav.numpy(), args.sample_rate, output_dir / "waveform.png")
    save_feature_plot(fbank.numpy(), "FBank", output_dir / "fbank.png")
    save_feature_plot(mfcc.numpy(), "MFCC", output_dir / "mfcc.png")
    save_feature_plot(fbank_cmvn.numpy(), "FBank + CMVN", output_dir / "fbank_cmvn.png")
    save_feature_plot(mfcc_cmvn.numpy(), "MFCC + CMVN", output_dir / "mfcc_cmvn.png")

    summary = {
        "input_wav": str(input_wav),
        "sample_rate": args.sample_rate,
        "num_samples": int(wav.numel()),
        "duration_sec": float(wav.numel() / args.sample_rate),
        "num_mel_bins": args.num_mel_bins,
        "num_ceps": args.num_ceps,
        "frame_length_ms": args.frame_length_ms,
        "frame_shift_ms": args.frame_shift_ms,
        "variance_norm": bool(args.variance_norm),
        "fbank_shape": list(fbank.shape),
        "mfcc_shape": list(mfcc.shape),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved frontend feature outputs to: {output_dir}")
    print("Generated files:")
    for name in [
        "waveform.png",
        "fbank.npy",
        "fbank.png",
        "mfcc.npy",
        "mfcc.png",
        "fbank_cmvn.npy",
        "fbank_cmvn.png",
        "mfcc_cmvn.npy",
        "mfcc_cmvn.png",
        "summary.json",
    ]:
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    main()
