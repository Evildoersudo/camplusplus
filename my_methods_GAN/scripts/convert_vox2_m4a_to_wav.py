#!/usr/bin/env python3
"""Convert VoxCeleb2 m4a files to a flattened wav tree.

Source layout:
  vox2/dev/aac/<speaker>/<video>/<utt>.m4a

Output layout:
  vox2_wav/wav/<speaker>/<video>/<utt>.wav

By default the output is mono 16 kHz WAV, which is the common speaker
verification format. M4A/AAC is compressed and has no PCM bit depth; the default
WAV codec is 16-bit PCM for broad Kaldi/tooling compatibility.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("egs/3dspeaker/sv-cam++/data/raw_data/vox2")
DEFAULT_OUTPUT_ROOT = Path("egs/3dspeaker/sv-cam++/data/raw_data/vox2_wav/wav")


@dataclass(frozen=True)
class ConvertTask:
    source: Path
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert VoxCeleb2 .m4a files to .wav while preserving the "
            "speaker/video/utterance layout under a flat wav root."
        )
    )
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset",
        choices=("dev", "test"),
        default="dev",
        help="VoxCeleb2 subset under INPUT_ROOT.",
    )
    parser.add_argument(
        "--sample_rate",
        "--sample-rate",
        type=int,
        default=16000,
        help="Output sample rate. Default: 16000 Hz.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Output channel count. Default: 1 mono.",
    )
    parser.add_argument(
        "--pcm_codec",
        "--pcm-codec",
        default="pcm_s16le",
        choices=("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le"),
        help="Output WAV PCM codec. Default: pcm_s16le.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="After conversion, verify sample_rate and channel count match requested output.",
    )
    return parser.parse_args()


def run(command: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-12:])
        raise RuntimeError(f"Command failed:\n{' '.join(command)}\n{tail}") from exc


def ffprobe_audio(path: Path) -> dict[str, str]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_fmt,sample_rate,channels,bits_per_sample,bits_per_raw_sample",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No audio stream found: {path}")
    return {str(k): str(v) for k, v in streams[0].items()}


def collect_tasks(input_root: Path, output_root: Path, dataset: str) -> list[ConvertTask]:
    source_root = input_root / dataset / "aac"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing VoxCeleb2 m4a root: {source_root}")

    tasks: list[ConvertTask] = []
    for source in sorted(source_root.glob("*/*/*.m4a")):
        relative = source.relative_to(source_root).with_suffix(".wav")
        tasks.append(ConvertTask(source=source, output=output_root / relative))

    if not tasks:
        raise RuntimeError(f"No .m4a files found under {source_root}")
    return tasks


def verify_requested_format(output: Path, sample_rate: int, channels: int) -> None:
    dst = ffprobe_audio(output)
    if str(sample_rate) != dst.get("sample_rate"):
        raise RuntimeError(
            f"sample_rate mismatch for {output}: expected={sample_rate} output={dst.get('sample_rate')}"
        )
    if str(channels) != dst.get("channels"):
        raise RuntimeError(
            f"channels mismatch for {output}: expected={channels} output={dst.get('channels')}"
        )


def convert_one(
    task: ConvertTask,
    pcm_codec: str,
    sample_rate: int,
    channels: int,
    overwrite: bool,
    strict: bool,
) -> None:
    if task.output.exists() and task.output.stat().st_size > 44 and not overwrite:
        return

    task.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(task.source),
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        pcm_codec,
        str(task.output),
    ]
    run(command)

    if strict:
        verify_requested_format(task.output, sample_rate, channels)


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be available in PATH.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    tasks = collect_tasks(input_root, output_root, args.dataset)

    first_info = ffprobe_audio(tasks[0].source)
    print(f"Input root:       {input_root / args.dataset / 'aac'}")
    print(f"Output root:      {output_root}")
    print(f"Files:            {len(tasks)}")
    print(f"First source:     {tasks[0].source}")
    print(
        "First stream:     "
        f"codec={first_info.get('codec_name')} "
        f"fmt={first_info.get('sample_fmt')} "
        f"sr={first_info.get('sample_rate')} "
        f"ch={first_info.get('channels')} "
        f"bits={first_info.get('bits_per_sample', 'N/A')}"
    )
    print(f"Output format:    {args.sample_rate} Hz, channels={args.channels}, codec={args.pcm_codec}")

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                convert_one,
                task,
                args.pcm_codec,
                args.sample_rate,
                args.channels,
                args.overwrite,
                args.strict,
            ): task
            for task in tasks
        }
        total = len(futures)
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to convert {task.source}") from exc
            if completed % 1000 == 0 or completed == total:
                print(f"Converted:        {completed}/{total}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
