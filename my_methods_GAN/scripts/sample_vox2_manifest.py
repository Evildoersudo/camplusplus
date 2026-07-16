#!/usr/bin/env python3
"""Sample each VoxCeleb2 speaker while preserving video/session diversity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from collections import defaultdict, deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep a fraction of every speaker in a clean manifest. Files are "
            "selected round-robin across videos to retain session diversity."
        )
    )
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--min_keep_per_speaker", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def infer_video_id(row: dict[str, str]) -> str:
    if row.get("video_id"):
        return row["video_id"]
    rel = row.get("rel_path") or row.get("clean_wav") or ""
    parts = Path(rel).parts
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"


def sample_speaker(
    speaker: str,
    rows: list[dict[str, str]],
    fraction: float,
    min_keep: int,
    seed: int,
) -> list[dict[str, str]]:
    target = min(len(rows), max(min_keep, math.ceil(len(rows) * fraction)))
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video[infer_video_id(row)].append(row)

    rng = random.Random(stable_seed(seed, speaker))
    video_ids = sorted(by_video)
    rng.shuffle(video_ids)
    queues: dict[str, deque[dict[str, str]]] = {}
    for video_id in video_ids:
        items = by_video[video_id][:]
        rng.shuffle(items)
        queues[video_id] = deque(items)

    selected: list[dict[str, str]] = []
    while len(selected) < target:
        made_progress = False
        for video_id in video_ids:
            queue = queues[video_id]
            if queue and len(selected) < target:
                selected.append(queue.popleft())
                made_progress = True
        if not made_progress:
            break
    return selected


def main() -> int:
    args = parse_args()
    if not (0.0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in (0, 1]")
    if args.min_keep_per_speaker < 1:
        raise ValueError("--min_keep_per_speaker must be at least 1")

    input_manifest = args.input_manifest.resolve()
    with input_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found: {input_manifest}")
    for required in ("spk_id", "clean_wav"):
        if required not in fields:
            raise ValueError(f"Manifest must contain {required!r}: {input_manifest}")

    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["spk_id"]].append(row)

    selected: list[dict[str, str]] = []
    for speaker in sorted(by_speaker):
        selected.extend(
            sample_speaker(
                speaker,
                by_speaker[speaker],
                args.fraction,
                args.min_keep_per_speaker,
                args.seed,
            )
        )
    random.Random(args.seed).shuffle(selected)

    output_manifest = args.output_manifest.resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    selected_counts: dict[str, int] = defaultdict(int)
    selected_videos: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        selected_counts[row["spk_id"]] += 1
        selected_videos[row["spk_id"]].add(infer_video_id(row))

    counts = list(selected_counts.values())
    print(f"Input manifest:       {input_manifest}")
    print(f"Output manifest:      {output_manifest}")
    print(f"Rows:                 {len(rows)} -> {len(selected)} ({len(selected) / len(rows):.4f})")
    print(f"Speakers retained:    {len(selected_counts)}/{len(by_speaker)}")
    print(
        "Rows per speaker:     "
        f"min={min(counts)} max={max(counts)} avg={sum(counts) / len(counts):.2f}"
    )
    print(f"Videos retained:      {sum(len(v) for v in selected_videos.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
