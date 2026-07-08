#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split pair manifest into train/valid by speaker id.")
    p.add_argument("--input_manifest", type=str, required=True)
    p.add_argument("--train_manifest", type=str, required=True)
    p.add_argument("--valid_manifest", type=str, required=True)
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of train rows to keep.")
    p.add_argument("--valid_fraction", type=float, default=1.0, help="Fraction of valid rows to keep.")
    p.add_argument("--stratified", action="store_true", help="Enable speaker-stratified row sampling after split.")
    p.add_argument(
        "--target_total_fraction",
        type=float,
        default=None,
        help=(
            "Optional total-row fraction to keep before train/valid split. "
            "When used with --max_rows_per_speaker, the script selects only "
            "enough speakers to approximately reach this total size."
        ),
    )
    p.add_argument(
        "--speaker_fraction",
        type=float,
        default=1.0,
        help="Optional fraction of eligible speakers to keep before train/valid split.",
    )
    p.add_argument(
        "--max_speakers",
        type=int,
        default=None,
        help="Optional maximum number of eligible speakers to keep before train/valid split.",
    )
    p.add_argument(
        "--min_rows_per_speaker",
        type=int,
        default=1,
        help="Drop speakers with fewer than this many rows before speaker selection.",
    )
    p.add_argument(
        "--max_rows_per_speaker",
        type=int,
        default=None,
        help="Optional cap on rows kept per selected speaker before train/valid split.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _sample_rows(rows: list[dict], fraction: float, seed: int, stratified: bool) -> list[dict]:
    if fraction >= 1.0:
        return rows
    if fraction <= 0.0:
        raise ValueError("fraction must be in (0, 1].")

    rng = random.Random(seed)
    if not stratified:
        sampled = rows[:]
        rng.shuffle(sampled)
        keep = max(1, int(len(sampled) * fraction))
        return sampled[:keep]

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["spk_id"]].append(row)

    sampled: list[dict] = []
    for spk, items in buckets.items():
        local = items[:]
        rng.shuffle(local)
        keep = max(1, int(len(local) * fraction))
        sampled.extend(local[:keep])

    rng.shuffle(sampled)
    return sampled


def _build_speaker_buckets(rows: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["spk_id"]].append(row)
    return buckets


def _select_speaker_subset(
    buckets: dict[str, list[dict]],
    total_rows: int,
    target_total_fraction: float | None,
    speaker_fraction: float,
    max_speakers: int | None,
    min_rows_per_speaker: int,
    max_rows_per_speaker: int | None,
    seed: int,
) -> tuple[list[dict], list[str], list[str]]:
    if min_rows_per_speaker < 1:
        raise ValueError("--min_rows_per_speaker must be at least 1.")
    if max_rows_per_speaker is not None and max_rows_per_speaker < 1:
        raise ValueError("--max_rows_per_speaker must be at least 1.")
    if not (0.0 < speaker_fraction <= 1.0):
        raise ValueError("--speaker_fraction must be in (0, 1].")
    if target_total_fraction is not None and not (0.0 < target_total_fraction <= 1.0):
        raise ValueError("--target_total_fraction must be in (0, 1].")
    if max_speakers is not None and max_speakers < 1:
        raise ValueError("--max_speakers must be at least 1.")

    eligible_spks = sorted(
        spk for spk, items in buckets.items() if len(items) >= min_rows_per_speaker
    )
    dropped_spks = sorted(set(buckets) - set(eligible_spks))
    if not eligible_spks:
        raise RuntimeError(
            "No speakers remain after --min_rows_per_speaker="
            f"{min_rows_per_speaker}."
        )

    rng = random.Random(seed)
    rng.shuffle(eligible_spks)

    desired_speakers = len(eligible_spks)
    if speaker_fraction < 1.0:
        desired_speakers = min(
            desired_speakers,
            max(1, int(len(eligible_spks) * speaker_fraction)),
        )
    if max_speakers is not None:
        desired_speakers = min(desired_speakers, max_speakers)
    if target_total_fraction is not None and max_rows_per_speaker is not None:
        target_rows = max(1, int(total_rows * target_total_fraction))
        speakers_for_target = max(1, (target_rows + max_rows_per_speaker - 1) // max_rows_per_speaker)
        desired_speakers = min(desired_speakers, speakers_for_target)

    selected_spks = eligible_spks[:desired_speakers]
    selected_rows: list[dict] = []
    for spk in selected_spks:
        local = buckets[spk][:]
        rng.shuffle(local)
        if max_rows_per_speaker is not None:
            local = local[:max_rows_per_speaker]
        selected_rows.extend(local)

    rng.shuffle(selected_rows)
    return selected_rows, selected_spks, dropped_spks


def main():
    args = parse_args()
    input_manifest = Path(args.input_manifest).resolve()
    reader = csv.DictReader(input_manifest.open("r", encoding="utf-8", newline=""))
    rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found: {input_manifest}")
    if not reader.fieldnames or "spk_id" not in reader.fieldnames:
        raise ValueError(f"Manifest must contain spk_id column: {input_manifest}")

    original_rows = len(rows)
    original_buckets = _build_speaker_buckets(rows)
    rows, selected_spks, dropped_spks = _select_speaker_subset(
        original_buckets,
        total_rows=original_rows,
        target_total_fraction=args.target_total_fraction,
        speaker_fraction=args.speaker_fraction,
        max_speakers=args.max_speakers,
        min_rows_per_speaker=args.min_rows_per_speaker,
        max_rows_per_speaker=args.max_rows_per_speaker,
        seed=args.seed,
    )

    spks = sorted({row["spk_id"] for row in rows})
    random.Random(args.seed).shuffle(spks)
    n_valid = max(1, int(len(spks) * args.valid_ratio))
    if len(spks) <= 1:
        raise RuntimeError("Need at least 2 selected speakers for speaker-level train/valid split.")
    n_valid = min(n_valid, len(spks) - 1)
    valid_spk = set(spks[:n_valid])

    train_rows = [row for row in rows if row["spk_id"] not in valid_spk]
    valid_rows = [row for row in rows if row["spk_id"] in valid_spk]

    train_rows = _sample_rows(train_rows, args.train_fraction, args.seed, args.stratified)
    valid_rows = _sample_rows(valid_rows, args.valid_fraction, args.seed + 1, args.stratified)

    fields = list(reader.fieldnames)

    train_manifest = Path(args.train_manifest).resolve()
    valid_manifest = Path(args.valid_manifest).resolve()
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    valid_manifest.parent.mkdir(parents=True, exist_ok=True)

    with train_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in train_rows])

    with valid_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in valid_rows])

    selected_counts = _build_speaker_buckets(rows)
    kept_per_spk = [len(items) for items in selected_counts.values()]
    print(f"rows original={original_rows} selected_before_split={len(rows)}")
    print(
        "speakers original={} eligible={} selected={} dropped_by_min_rows={} valid={}".format(
            len(original_buckets),
            len(original_buckets) - len(dropped_spks),
            len(selected_spks),
            len(dropped_spks),
            len(valid_spk),
        )
    )
    print(
        "selected rows per speaker: min={} max={} avg={:.2f}".format(
            min(kept_per_spk),
            max(kept_per_spk),
            sum(kept_per_spk) / len(kept_per_spk),
        )
    )
    print(f"train rows={len(train_rows)} (fraction={args.train_fraction}, stratified={args.stratified})")
    print(f"valid rows={len(valid_rows)} (fraction={args.valid_fraction}, stratified={args.stratified})")
    print(f"saved train: {train_manifest}")
    print(f"saved valid: {valid_manifest}")


if __name__ == "__main__":
    main()
