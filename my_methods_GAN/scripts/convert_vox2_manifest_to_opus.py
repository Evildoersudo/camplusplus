#!/usr/bin/env python3
"""Encode only manifest-selected VoxCeleb2 M4A files to Opus.

Each input clean manifest has a corresponding output pair manifest:
  utt_id,spk_id,clean_wav,codec_wav,codec_type
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    source: Path
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--pair_manifests_out", type=Path, nargs="+", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--bitrate", default="16k")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--application", choices=("voip", "audio", "lowdelay"), default="voip")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="After conversion, verify a deterministic sample of Opus outputs with ffprobe.",
    )
    parser.add_argument(
        "--strict_samples",
        type=int,
        default=100,
        help="Number of outputs checked by --strict; use 0 to verify every output. Default: 100.",
    )
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
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
        tail = "\n".join((exc.stderr or "").strip().splitlines()[-12:])
        raise RuntimeError(f"Command failed:\n{' '.join(command)}\n{tail}") from exc


def ffprobe(path: Path) -> dict[str, str]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"No audio stream: {path}")
    return {str(k): str(v) for k, v in streams[0].items()}


def infer_relative_path(row: dict[str, str], source: Path) -> Path:
    if row.get("rel_path"):
        rel = Path(row["rel_path"])
        if rel.is_absolute():
            raise ValueError(f"rel_path must be relative: {rel}")
    else:
        if len(source.parts) < 3:
            raise ValueError(f"Cannot infer speaker/video/file path from {source}")
        rel = Path(*source.parts[-3:])
    if len(rel.parts) != 3:
        raise ValueError(f"Expected speaker/video/file relative path, got: {rel}")
    return rel.with_suffix(".opus")


def read_manifest(path: Path, output_root: Path) -> tuple[list[str], list[dict[str, str]], list[Task]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found: {path}")
    for required in ("utt_id", "spk_id", "clean_wav"):
        if required not in fields:
            raise ValueError(f"Manifest must contain {required!r}: {path}")

    tasks: list[Task] = []
    for row in rows:
        source = Path(row["clean_wav"]).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing clean audio: {source}")
        relative = infer_relative_path(row, source)
        row["_codec_wav"] = str(output_root / relative)
        tasks.append(Task(source, output_root / relative))
    return fields, rows, tasks


def convert_one(
    task: Task,
    bitrate: str,
    sample_rate: int,
    channels: int,
    application: str,
    overwrite: bool,
) -> None:
    if task.output.exists() and task.output.stat().st_size > 0 and not overwrite:
        return
    task.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = task.output.with_suffix(task.output.suffix + ".tmp")
    command = [
        "ffmpeg",
        "-nostdin",
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
        "libopus",
        "-b:a",
        bitrate,
        "-application",
        application,
        "-vbr",
        "on",
        "-threads",
        "1",
        "-f",
        "opus",
        str(temporary),
    ]
    try:
        run(command)
        temporary.replace(task.output)
    finally:
        if temporary.exists():
            temporary.unlink()

def run_bounded(tasks: list[Task], worker, workers: int) -> None:
    iterator = iter(tasks)
    pending: deque[tuple[concurrent.futures.Future, Task]] = deque()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(min(len(tasks), workers * 4)):
            task = next(iterator, None)
            if task is None:
                break
            pending.append((pool.submit(worker, task), task))

        while pending:
            future, task = pending.popleft()
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to encode {task.source}") from exc
            completed += 1
            if completed % 1000 == 0 or completed == len(tasks):
                print(f"Encoded/checked:      {completed}/{len(tasks)}")
            next_task = next(iterator, None)
            if next_task is not None:
                pending.append((pool.submit(worker, next_task), next_task))


def select_verification_tasks(tasks: list[Task], sample_count: int) -> list[Task]:
    if sample_count < 0:
        raise ValueError("--strict_samples must be at least 0")
    if sample_count == 0 or sample_count >= len(tasks):
        return tasks
    if sample_count == 1:
        return [tasks[0]]
    last = len(tasks) - 1
    indexes = sorted({round(index * last / (sample_count - 1)) for index in range(sample_count)})
    return [tasks[index] for index in indexes]


def verify_outputs(tasks: list[Task], channels: int, workers: int) -> None:
    def verify_one(task: Task) -> None:
        info = ffprobe(task.output)
        if info.get("codec_name") != "opus":
            raise RuntimeError(f"Expected Opus output, got {info}: {task.output}")
        if info.get("channels") != str(channels):
            raise RuntimeError(
                f"Channel mismatch, expected {channels}, got {info}: {task.output}"
            )

    run_bounded(tasks, verify_one, workers)


def write_pair_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utt_id", "spk_id", "clean_wav", "codec_wav", "codec_type"])
        for row in rows:
            writer.writerow(
                [row["utt_id"], row["spk_id"], row["clean_wav"], row["_codec_wav"], "opus"]
            )


def main() -> int:
    args = parse_args()
    if len(args.input_manifests) != len(args.pair_manifests_out):
        raise ValueError("--input_manifests and --pair_manifests_out must have equal lengths")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.strict_samples < 0:
        raise ValueError("--strict_samples must be at least 0")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be available in PATH")

    output_root = args.output_root.resolve()
    manifest_data: list[tuple[Path, list[dict[str, str]]]] = []
    unique_tasks: dict[Path, Task] = {}
    for input_path, output_path in zip(args.input_manifests, args.pair_manifests_out):
        _, rows, tasks = read_manifest(input_path.resolve(), output_root)
        manifest_data.append((output_path.resolve(), rows))
        for task in tasks:
            previous = unique_tasks.get(task.output)
            if previous is not None and previous.source != task.source:
                raise RuntimeError(f"Output collision: {previous.source} and {task.source} -> {task.output}")
            unique_tasks[task.output] = task

    tasks = sorted(unique_tasks.values(), key=lambda item: str(item.output))
    print(f"Input manifests:      {len(args.input_manifests)}")
    print(f"Selected utterances:  {sum(len(rows) for _, rows in manifest_data)}")
    print(f"Unique Opus outputs:  {len(tasks)}")
    print(f"Output root:          {output_root}")
    print(f"Opus config:          bitrate={args.bitrate}, sr={args.sample_rate}, channels={args.channels}")
    if tasks:
        print(f"First pair:           {tasks[0].source} -> {tasks[0].output}")

    if args.dry_run:
        print("Dry run complete; no Opus files or pair manifests were written.")
        return 0

    worker = lambda task: convert_one(
        task,
        args.bitrate,
        args.sample_rate,
        args.channels,
        args.application,
        args.overwrite,
    )
    run_bounded(tasks, worker, args.workers)

    if args.strict:
        verification_tasks = select_verification_tasks(tasks, args.strict_samples)
        print(f"Strict verification:  {len(verification_tasks)}/{len(tasks)} outputs")
        verify_outputs(verification_tasks, args.channels, min(args.workers, 8))

    for output_path, rows in manifest_data:
        write_pair_manifest(output_path, rows)
        print(f"Pair manifest:        {output_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
