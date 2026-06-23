#!/usr/bin/env python3
"""Build clean/coded true-paired VoxCeleb1 development training data.

The VoxCeleb1 test speakers found in the official verification trials are
always excluded, preventing train/test speaker leakage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path


DEFAULT_WAV_ROOT = Path("/root/autodl-tmp/raw_data/vox1/train/wav")
DEFAULT_TRIALS = Path(
    "/root/autodl-tmp/raw_data/vox1/data/vox1_trials/veri_test2.txt"
)
DEFAULT_OUTPUT = Path("/root/autodl-tmp/SC_data/data/voxceleb1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build clean/coded true-paired VoxCeleb1 training WAVs while "
            "excluding all official test speakers."
        )
    )
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_WAV_ROOT)
    parser.add_argument(
        "--test_trials",
        type=Path,
        default=DEFAULT_TRIALS,
        help="Official VoxCeleb trial list used to identify test speakers.",
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--codec",
        choices=("opus", "aac", "amrwb", "g711_mulaw", "g711_alaw"),
        default="opus",
    )
    parser.add_argument("--bitrate", default="16k")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Validate and report the split without writing audio.",
    )
    return parser.parse_args()


def load_test_speakers(trials_path: Path) -> set[str]:
    speakers: set[str] = set()
    with trials_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"Malformed trial line {line_number} in {trials_path}: {line!r}"
                )
            if fields[0].lower() in {"0", "1", "target", "nontarget"}:
                utterances = fields[1:]
            else:
                utterances = fields[:2]
            for utterance in utterances:
                parts = Path(utterance).parts
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid utterance on line {line_number}: {utterance!r}"
                    )
                speakers.add(parts[0])
    if not speakers:
        raise ValueError(f"No test speakers found in {trials_path}")
    return speakers


def run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        tail = "\n".join(detail[-10:])
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(command)}\n{tail}") from exc


def normalize_clean(
    source: Path,
    destination: Path,
    sample_rate: int,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-i", str(source), "-vn",
            "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
            str(destination),
        ]
    )


def build_codec_commands(
    clean_wav: Path,
    coded_wav: Path,
    codec: str,
    bitrate: str,
    sample_rate: int,
) -> tuple[Path, list[str], list[str]]:
    if codec == "opus":
        temporary = coded_wav.with_suffix(".opus")
        encode = [
            "ffmpeg", "-y", "-i", str(clean_wav), "-vn",
            "-c:a", "libopus", "-b:a", bitrate, str(temporary),
        ]
    elif codec == "aac":
        temporary = coded_wav.with_suffix(".m4a")
        encode = [
            "ffmpeg", "-y", "-i", str(clean_wav), "-vn",
            "-ar", "16000", "-ac", "1", "-c:a", "aac", "-b:a", bitrate,
            str(temporary),
        ]
    elif codec == "amrwb":
        temporary = coded_wav.with_suffix(".amr")
        encode = [
            "ffmpeg", "-y", "-i", str(clean_wav), "-vn",
            "-ar", "16000", "-ac", "1", "-c:a", "libvo_amrwbenc",
            "-b:a", bitrate, "-f", "amr", str(temporary),
        ]
    elif codec == "g711_mulaw":
        temporary = coded_wav.with_name(coded_wav.stem + ".mulaw.wav")
        encode = [
            "ffmpeg", "-y", "-i", str(clean_wav), "-vn",
            "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(temporary),
        ]
    else:
        temporary = coded_wav.with_name(coded_wav.stem + ".alaw.wav")
        encode = [
            "ffmpeg", "-y", "-i", str(clean_wav), "-vn",
            "-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw", str(temporary),
        ]

    decode = [
        "ffmpeg", "-y", "-i", str(temporary), "-vn",
        "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
        str(coded_wav),
    ]
    return temporary, encode, decode


def create_coded(
    clean_wav: Path,
    coded_wav: Path,
    codec: str,
    bitrate: str,
    sample_rate: int,
    overwrite: bool,
) -> None:
    if coded_wav.exists() and not overwrite:
        return
    coded_wav.parent.mkdir(parents=True, exist_ok=True)
    temporary, encode, decode = build_codec_commands(
        clean_wav, coded_wav, codec, bitrate, sample_rate
    )
    try:
        run_ffmpeg(encode)
        run_ffmpeg(decode)
    finally:
        temporary.unlink(missing_ok=True)


def run_parallel(tasks: list[tuple], function, workers: int, description: str) -> None:
    if not tasks:
        print(f"{description}: no tasks")
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(function, *task): task for task in tasks}
        total = len(futures)
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                source = futures[future][0]
                raise RuntimeError(f"{description} failed for {source}") from exc
            if completed % 1000 == 0 or completed == total:
                print(f"{description}: {completed}/{total}")


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    trials_path = args.test_trials.resolve()
    output_root = args.output_root.resolve()

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing VoxCeleb WAV root: {raw_root}")
    if not trials_path.is_file():
        raise FileNotFoundError(f"Missing official test trials: {trials_path}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    test_speakers = load_test_speakers(trials_path)
    all_wavs = sorted(raw_root.glob("*/*/*.wav"))
    if not all_wavs:
        raise RuntimeError(f"No id/video/*.wav files found under {raw_root}")

    all_speakers = {path.relative_to(raw_root).parts[0] for path in all_wavs}
    overlap = all_speakers & test_speakers
    train_wavs = [
        path
        for path in all_wavs
        if path.relative_to(raw_root).parts[0] not in test_speakers
    ]
    train_speakers = {
        path.relative_to(raw_root).parts[0] for path in train_wavs
    }

    if overlap != test_speakers:
        missing = sorted(test_speakers - all_speakers)
        raise RuntimeError(
            "Official test speakers do not exactly match the WAV tree; "
            f"missing test speakers: {missing[:10]}"
        )
    if train_speakers & test_speakers:
        raise RuntimeError("Train/test speaker leakage detected")

    bitrate_tag = args.bitrate.replace(".", "")
    clean_root = output_root / "clean_train_wav"
    coded_root = output_root / f"coded_train_{args.codec}_{bitrate_tag}"

    print(f"Source WAVs:          {len(all_wavs)}")
    print(f"All speakers:         {len(all_speakers)}")
    print(f"Excluded test WAVs:   {len(all_wavs) - len(train_wavs)}")
    print(f"Excluded test spkrs:  {len(test_speakers)}")
    print(f"Training WAVs:        {len(train_wavs)}")
    print(f"Training speakers:    {len(train_speakers)}")
    print(f"Clean output:         {clean_root}")
    print(f"Coded output:         {coded_root}")

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    clean_tasks = []
    coded_tasks = []
    for source in train_wavs:
        relative = source.relative_to(raw_root)
        clean_wav = clean_root / relative
        coded_wav = coded_root / relative
        clean_tasks.append((source, clean_wav, args.sample_rate, args.overwrite))
        coded_tasks.append(
            (
                clean_wav,
                coded_wav,
                args.codec,
                args.bitrate,
                args.sample_rate,
                args.overwrite,
            )
        )

    run_parallel(clean_tasks, normalize_clean, args.workers, "train clean")
    run_parallel(coded_tasks, create_coded, args.workers, "train coded")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
