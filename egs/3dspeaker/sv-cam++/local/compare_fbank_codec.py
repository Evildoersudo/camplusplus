import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torchaudio

try:
    from speakerlab.process.processor import FBank
except ImportError:
    # Allow running this script directly via "python path/to/script.py" by
    # adding the project root to sys.path.
    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from speakerlab.process.processor import FBank


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got: {value}")
    return parsed


def parse_args():
    # This script compares the log-mel filterbank features of a clean utterance
    # and its codec-degraded counterpart. It saves both the time-frequency
    # images and the average mel-bin energy curves so high-frequency truncation
    # can be inspected visually.
    parser = argparse.ArgumentParser(
        description="Compare clean and codec-degraded FBank features and save a visualization."
    )
    parser.add_argument("--clean_wav", type=str, required=True, help="Path to the clean wav file")
    parser.add_argument("--degraded_wav", type=str, required=True, help="Path to the degraded wav file")
    parser.add_argument("--out_png", type=str, required=True, help="Path to the output figure")
    parser.add_argument("--sample_rate", type=positive_int, default=16000, help="Target sample rate")
    parser.add_argument("--n_mels", type=positive_int, default=80, help="Number of mel bins")
    parser.add_argument(
        "--max_frames",
        type=positive_int,
        default=400,
        help="Maximum number of frames to plot for each utterance",
    )
    return parser.parse_args()


def _format_backend_errors(errors: List[str]) -> str:
    if not errors:
        return ""
    return "\n".join(f"  - {item}" for item in errors)


def _torchaudio_load_with_fallback(wav_path: Path):
    backend_errors: List[str] = []

    # Prefer ffmpeg/sox if available because some codec-derived wav files are
    # readable there but fail in soundfile with an opaque "System error".
    try:
        available = torchaudio.list_audio_backends()
    except Exception:
        available = []

    preferred = ["ffmpeg", "sox_io", "soundfile"]
    ordered_backends = [b for b in preferred if b in available]
    ordered_backends += [b for b in available if b not in ordered_backends]

    if not ordered_backends:
        return torchaudio.load(str(wav_path))

    for backend in ordered_backends:
        try:
            return torchaudio.load(str(wav_path), backend=backend)
        except TypeError:
            # Older torchaudio versions may not support backend kwarg.
            return torchaudio.load(str(wav_path))
        except Exception as exc:
            backend_errors.append(f"{backend}: {exc}")

    raise RuntimeError(
        f"Failed to decode audio after trying backends: {ordered_backends}\n"
        f"File: {wav_path}\n"
        f"Backend errors:\n{_format_backend_errors(backend_errors)}"
    )


def load_wav(wav_path: Path, sample_rate: int):
    if not wav_path.exists():
        raise FileNotFoundError(
            f"Wav file not found: {wav_path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    try:
        wav, sr = _torchaudio_load_with_fallback(wav_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read wav file: {wav_path}\n"
            "Please verify: (1) path is correct, (2) file is not locked/corrupted, "
            "(3) codec is decodable by your torchaudio/ffmpeg installation."
        ) from exc

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[0] > 1:
        wav = wav[:1]
    return wav


def compute_fbank(wav: torch.Tensor, sample_rate: int, n_mels: int):
    extractor = FBank(n_mels=n_mels, sample_rate=sample_rate, mean_nor=True)
    feat = extractor(wav.squeeze(0))
    return feat


def trim_feat(feat: torch.Tensor, max_frames: int):
    if feat.shape[0] <= max_frames:
        return feat
    return feat[:max_frames]


def plot_results(clean_feat: torch.Tensor, degraded_feat: torch.Tensor, out_png: Path, clean_wav: Path, degraded_wav: Path):
    clean_img = clean_feat.transpose(0, 1).cpu().numpy()
    degraded_img = degraded_feat.transpose(0, 1).cpu().numpy()
    clean_curve = clean_img.mean(axis=1)
    degraded_curve = degraded_img.mean(axis=1)
    diff_curve = degraded_curve - clean_curve

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].imshow(clean_img, aspect="auto", origin="lower")
    axes[0, 0].set_title(f"Clean FBank\n{clean_wav.name}")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("Mel bin")

    axes[0, 1].imshow(degraded_img, aspect="auto", origin="lower")
    axes[0, 1].set_title(f"Degraded FBank\n{degraded_wav.name}")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Mel bin")

    axes[1, 0].plot(clean_curve, label="clean")
    axes[1, 0].plot(degraded_curve, label="degraded")
    axes[1, 0].set_title("Average energy per mel bin")
    axes[1, 0].set_xlabel("Mel bin")
    axes[1, 0].set_ylabel("Mean log-mel energy")
    axes[1, 0].legend()

    axes[1, 1].plot(diff_curve, color="tab:red")
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set_title("Degraded - Clean mel-bin difference")
    axes[1, 1].set_xlabel("Mel bin")
    axes[1, 1].set_ylabel("Energy difference")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    clean_wav = Path(args.clean_wav).resolve()
    degraded_wav = Path(args.degraded_wav).resolve()
    out_png = Path(args.out_png).resolve()

    clean_signal = load_wav(clean_wav, args.sample_rate)
    degraded_signal = load_wav(degraded_wav, args.sample_rate)

    clean_feat = trim_feat(compute_fbank(clean_signal, args.sample_rate, args.n_mels), args.max_frames)
    degraded_feat = trim_feat(compute_fbank(degraded_signal, args.sample_rate, args.n_mels), args.max_frames)

    plot_results(clean_feat, degraded_feat, out_png, clean_wav, degraded_wav)
    print(f"Saved FBank comparison figure to: {out_png}")


if __name__ == "__main__":
    main()
