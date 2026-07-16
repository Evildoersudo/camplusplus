#!/usr/bin/env python3
"""Build clean/coded true-paired training data.

For VoxCeleb1, pass --exclude_test_speakers with --test_trials to exclude
official verification speakers and prevent train/test speaker leakage.
For datasets such as LibriSpeech, leave it unset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from sv_codec_restore_gan.data.manifest import build_pair_manifest
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    data_dir = Path(__file__).resolve().parents[1] / "sv_codec_restore_gan" / "data"
    sys.path.insert(0, str(data_dir))
    from manifest import build_pair_manifest


DEFAULT_WAV_ROOT = Path("/root/autodl-tmp/raw_data/vox1/train/wav")
DEFAULT_TRIALS = Path(
    "/root/autodl-tmp/raw_data/vox1/data/vox1_trials/veri_test2.txt"
)
DEFAULT_OUTPUT = Path("/root/autodl-tmp/SC_data/data/voxceleb1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build clean/coded true-paired training WAVs. VoxCeleb1 official "
            "test-speaker exclusion is optional."
        )
    )
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_WAV_ROOT)
    parser.add_argument(
        "--test_trials",
        type=Path,
        default=DEFAULT_TRIALS,
        help=(
            "Official VoxCeleb trial list used to identify test speakers. "
            "Only required when --exclude_test_speakers is set."
        ),
    )
    parser.add_argument(
        "--exclude_test_speakers",
        "--exclude-test-speakers",
        action="store_true",
        help=(
            "Exclude speakers found in --test_trials. Use this for VoxCeleb1; "
            "leave unset for independent datasets such as LibriSpeech."
        ),
    )
    parser.add_argument(
        "--test_wav_root",
        "--test-wav-root",
        type=Path,
        default=None,
        help=(
            "Optional test WAV root in id/video/*.wav layout. Only used with "
            "--exclude_test_speakers when --raw_root is already a pre-split "
            "training tree."
        ),
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--clean_root",
        "--clean-root",
        type=Path,
        default=None,
        help=(
            "Existing clean WAV root in id/video/*.wav layout. Defaults to "
            "OUTPUT_ROOT/clean_train_wav and is used to resume coded-only "
            "generation when --raw_root is unavailable."
        ),
    )
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
        "--clean_write_mode",
        "--clean-write-mode",
        choices=("normalize", "copy", "hardlink", "symlink"),
        default="normalize",
        help=(
            "How to create clean_train_wav when --raw_root is available. "
            "'normalize' keeps the legacy FFmpeg 16 kHz mono PCM rewrite; "
            "'copy', 'hardlink', or 'symlink' avoid clean->clean transcoding. "
            "For VoxCeleb1 train/wav that is already 16 kHz mono WAV, "
            "'hardlink' is usually fastest and uses no extra audio storage."
        ),
    )
    parser.add_argument(
        "--include_manifest",
        "--include-manifest",
        type=Path,
        default=None,
        help=(
            "Optional manifest that selects which utterances to process. "
            "Rows may contain clean_wav paths or rel_path values."
        ),
    )
    parser.add_argument(
        "--valid_include_manifest",
        "--valid-include-manifest",
        type=Path,
        default=None,
        help=(
            "Optional validation manifest. When provided, the script also "
            "transcodes this subset after the training subset."
        ),
    )
    parser.add_argument(
        "--train_pair_manifest",
        "--train-pair-manifest",
        type=Path,
        default=None,
        help="Optional output CSV for the train clean/coded pair manifest.",
    )
    parser.add_argument(
        "--valid_pair_manifest",
        "--valid-pair-manifest",
        type=Path,
        default=None,
        help="Optional output CSV for the valid clean/coded pair manifest.",
    )
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


def collect_wavs(wav_root: Path) -> list[Path]:
    return sorted(wav_root.glob("*/*/*.wav"))


def collect_speakers(wav_root: Path, wavs: list[Path]) -> set[str]:
    return {path.relative_to(wav_root).parts[0] for path in wavs}


def _relative_manifest_path(path_text: str, roots: list[Path]) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path.as_posix()
    for root in roots:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    if len(path.parts) >= 3:
        return Path(*path.parts[-3:]).as_posix()
    raise ValueError(f"Cannot infer id/video/file relative path from {path_text!r}")


def load_include_relpaths(manifest_path: Path, roots: list[Path]) -> set[str]:
    relpaths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        for row_number, row in enumerate(reader, 2):
            path_text = row.get("rel_path") or row.get("clean_wav")
            if not path_text:
                raise ValueError(
                    f"Manifest row {row_number} must contain rel_path or clean_wav"
                )
            relpaths.add(_relative_manifest_path(path_text, roots))
    if not relpaths:
        raise ValueError(f"No include rows found in {manifest_path}")
    return relpaths


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


def write_clean(
    source: Path,
    destination: Path,
    sample_rate: int,
    overwrite: bool,
    clean_write_mode: str,
) -> None:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if clean_write_mode == "normalize":
        normalize_clean(source, destination, sample_rate, overwrite=True)
    elif clean_write_mode == "copy":
        shutil.copy2(source, destination)
    elif clean_write_mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            # Hard links require source and destination to be on the same
            # filesystem. Fall back to copy so long runs do not fail late.
            shutil.copy2(source, destination)
    elif clean_write_mode == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(f"Unsupported clean_write_mode: {clean_write_mode}")


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
    if coded_wav.exists() and not overwrite and coded_wav.stat().st_size > 44:
        return
    coded_wav.parent.mkdir(parents=True, exist_ok=True)
    coded_wav.unlink(missing_ok=True)
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


def select_wavs(
    wavs: list[Path],
    source_root: Path,
    test_speakers: set[str],
    include_manifest: Path | None,
    clean_root: Path,
    raw_root: Path,
    label: str,
) -> list[Path]:
    selected = [
        path
        for path in wavs
        if path.relative_to(source_root).parts[0] not in test_speakers
    ]
    if include_manifest is None:
        return selected

    include_relpaths = load_include_relpaths(
        include_manifest,
        [source_root, clean_root, raw_root],
    )
    selected = [
        path
        for path in selected
        if path.relative_to(source_root).as_posix() in include_relpaths
    ]
    if not selected:
        raise RuntimeError(
            f"No source WAVs matched --{label}_include_manifest: {include_manifest}"
        )
    return selected


def build_tasks(
    wavs: list[Path],
    source_root: Path,
    clean_root: Path,
    coded_root: Path,
    coded_only: bool,
    sample_rate: int,
    overwrite: bool,
    codec: str,
    bitrate: str,
    clean_write_mode: str,
) -> tuple[list[tuple], list[tuple]]:
    clean_tasks = []
    coded_tasks = []
    for source in wavs:
        relative = source.relative_to(source_root)
        clean_wav = source if coded_only else clean_root / relative
        coded_wav = coded_root / relative
        if not coded_only:
            clean_tasks.append((source, clean_wav, sample_rate, overwrite, clean_write_mode))
        coded_tasks.append(
            (
                clean_wav,
                coded_wav,
                codec,
                bitrate,
                sample_rate,
                overwrite,
            )
        )
    return clean_tasks, coded_tasks


def write_pair_manifest(
    manifest_path: Path | None,
    clean_root: Path,
    coded_root: Path,
    include_manifest: Path | None,
    label: str,
) -> None:
    if manifest_path is None:
        return
    if include_manifest is None:
        raise ValueError(f"--{label}_pair_manifest requires --{label}_include_manifest")
    count = build_pair_manifest(
        clean_root,
        coded_root,
        manifest_path,
        include_manifest=include_manifest,
    )
    print(f"Saved {label} pair manifest: {manifest_path.resolve()}")
    print(f"{label.capitalize()} paired utterances: {count}")


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    trials_path = args.test_trials.resolve() if args.test_trials else None
    test_wav_root = args.test_wav_root.resolve() if args.test_wav_root else None
    include_manifest = args.include_manifest.resolve() if args.include_manifest else None
    valid_include_manifest = (
        args.valid_include_manifest.resolve()
        if args.valid_include_manifest
        else None
    )
    train_pair_manifest = (
        args.train_pair_manifest.resolve()
        if args.train_pair_manifest
        else None
    )
    valid_pair_manifest = (
        args.valid_pair_manifest.resolve()
        if args.valid_pair_manifest
        else None
    )
    output_root = args.output_root.resolve()
    bitrate_tag = args.bitrate.replace(".", "")
    clean_root = (
        args.clean_root.resolve()
        if args.clean_root
        else output_root / "clean_train_wav"
    )
    coded_root = output_root / f"coded_train_{args.codec}_{bitrate_tag}"
    raw_root_exists = raw_root.is_dir()
    clean_root_exists = clean_root.is_dir()
    coded_only = not raw_root_exists and clean_root_exists

    if not raw_root_exists and not clean_root_exists:
        raise FileNotFoundError(
            f"Missing source WAV root: {raw_root}; also missing clean WAV "
            f"root for coded-only resume: {clean_root}"
        )
    if args.exclude_test_speakers and (
        trials_path is None or not trials_path.is_file()
    ):
        raise FileNotFoundError(f"Missing official test trials: {trials_path}")
    if test_wav_root is not None and not test_wav_root.is_dir():
        raise FileNotFoundError(f"Missing test WAV root: {test_wav_root}")
    if include_manifest is not None and not include_manifest.is_file():
        raise FileNotFoundError(f"Missing include manifest: {include_manifest}")
    if valid_include_manifest is not None and not valid_include_manifest.is_file():
        raise FileNotFoundError(
            f"Missing valid include manifest: {valid_include_manifest}"
        )
    if train_pair_manifest is not None and include_manifest is None:
        raise ValueError("--train_pair_manifest requires --include_manifest")
    if valid_pair_manifest is not None and valid_include_manifest is None:
        raise ValueError("--valid_pair_manifest requires --valid_include_manifest")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    test_speakers = (
        load_test_speakers(trials_path) if args.exclude_test_speakers else set()
    )
    if args.exclude_test_speakers and test_wav_root is not None:
        test_wavs = collect_wavs(test_wav_root)
        if not test_wavs:
            raise RuntimeError(f"No id/video/*.wav files found under {test_wav_root}")
        test_tree_speakers = collect_speakers(test_wav_root, test_wavs)
        missing_test_speakers = sorted(test_speakers - test_tree_speakers)
        if missing_test_speakers:
            raise RuntimeError(
                "Official test speakers are missing from --test_wav_root; "
                f"missing test speakers: {missing_test_speakers[:10]}"
            )
    else:
        test_wavs = []
        test_tree_speakers = set()

    source_root = clean_root if coded_only else raw_root
    all_wavs = collect_wavs(source_root)
    if not all_wavs:
        raise RuntimeError(f"No id/video/*.wav files found under {source_root}")

    all_speakers = collect_speakers(source_root, all_wavs)
    overlap = all_speakers & test_speakers
    train_wavs = select_wavs(
        all_wavs,
        source_root,
        test_speakers,
        include_manifest,
        clean_root,
        raw_root,
        "train",
    )
    valid_wavs = (
        select_wavs(
            all_wavs,
            source_root,
            test_speakers,
            valid_include_manifest,
            clean_root,
            raw_root,
            "valid",
        )
        if valid_include_manifest is not None
        else []
    )

    train_relpaths = {path.relative_to(source_root).as_posix() for path in train_wavs}
    valid_relpaths = {path.relative_to(source_root).as_posix() for path in valid_wavs}
    overlap_relpaths = train_relpaths & valid_relpaths
    if overlap_relpaths:
        examples = sorted(overlap_relpaths)[:10]
        raise RuntimeError(
            "Train/valid include manifests overlap; "
            f"example overlapping utterances: {examples}"
        )
    train_speakers = {
        path.relative_to(source_root).parts[0] for path in train_wavs
    }
    valid_speakers = {
        path.relative_to(source_root).parts[0] for path in valid_wavs
    }

    if (
        args.exclude_test_speakers
        and test_wav_root is None
        and overlap != test_speakers
    ):
        missing = sorted(test_speakers - all_speakers)
        raise RuntimeError(
            "Official test speakers do not exactly match the WAV tree; "
            f"missing test speakers: {missing[:10]}. If --raw_root is already "
            "a pre-split training tree, pass --test_wav_root to validate the "
            "official test speakers separately."
        )
    if args.exclude_test_speakers and train_speakers & test_speakers:
        raise RuntimeError("Train/test speaker leakage detected")

    print(f"Mode:                 {'coded-only resume' if coded_only else 'raw-to-clean-and-coded'}")
    print(f"Source root:          {source_root}")
    print(f"Source WAVs:          {len(all_wavs)}")
    print(f"All speakers:         {len(all_speakers)}")
    if args.exclude_test_speakers:
        print(f"Test trials:          {trials_path}")
    else:
        print("Test speaker filter:  disabled")
    if args.exclude_test_speakers and test_wav_root is not None:
        print(f"Test WAV root:        {test_wav_root}")
        print(f"Test WAVs:            {len(test_wavs)}")
        print(f"Test tree speakers:   {len(test_tree_speakers)}")
    if include_manifest is not None:
        print(f"Train include:        {include_manifest}")
    if valid_include_manifest is not None:
        print(f"Valid include:        {valid_include_manifest}")
    print(f"Excluded test WAVs:   {len(all_wavs) - len(train_wavs) - len(valid_wavs)}")
    print(f"Excluded test spkrs:  {len(test_speakers)}")
    print(f"Training WAVs:        {len(train_wavs)}")
    print(f"Training speakers:    {len(train_speakers)}")
    if valid_include_manifest is not None:
        print(f"Validation WAVs:      {len(valid_wavs)}")
        print(f"Validation speakers:  {len(valid_speakers)}")
    print(f"Clean output:         {clean_root}")
    if not coded_only:
        print(f"Clean write mode:     {args.clean_write_mode}")
    print(f"Coded output:         {coded_root}")

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    clean_tasks, coded_tasks = build_tasks(
        train_wavs,
        source_root,
        clean_root,
        coded_root,
        coded_only,
        args.sample_rate,
        args.overwrite,
        args.codec,
        args.bitrate,
        args.clean_write_mode,
    )
    valid_clean_tasks, valid_coded_tasks = build_tasks(
        valid_wavs,
        source_root,
        clean_root,
        coded_root,
        coded_only,
        args.sample_rate,
        args.overwrite,
        args.codec,
        args.bitrate,
        args.clean_write_mode,
    )

    if not coded_only:
        run_parallel(clean_tasks, write_clean, args.workers, "train clean")
        run_parallel(valid_clean_tasks, write_clean, args.workers, "valid clean")
    run_parallel(coded_tasks, create_coded, args.workers, "train coded")
    run_parallel(valid_coded_tasks, create_coded, args.workers, "valid coded")
    write_pair_manifest(
        train_pair_manifest,
        clean_root,
        coded_root,
        include_manifest,
        "train",
    )
    write_pair_manifest(
        valid_pair_manifest,
        clean_root,
        coded_root,
        valid_include_manifest,
        "valid",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
