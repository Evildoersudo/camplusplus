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


def main():
    args = parse_args()
    input_manifest = Path(args.input_manifest).resolve()
    reader = csv.DictReader(input_manifest.open("r", encoding="utf-8", newline=""))
    rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found: {input_manifest}")
    if not reader.fieldnames or "spk_id" not in reader.fieldnames:
        raise ValueError(f"Manifest must contain spk_id column: {input_manifest}")

    spks = sorted({row["spk_id"] for row in rows})
    random.Random(args.seed).shuffle(spks)
    n_valid = max(1, int(len(spks) * args.valid_ratio))
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

    print(f"speakers total={len(spks)} valid={len(valid_spk)}")
    print(f"train rows={len(train_rows)} (fraction={args.train_fraction}, stratified={args.stratified})")
    print(f"valid rows={len(valid_rows)} (fraction={args.valid_fraction}, stratified={args.stratified})")
    print(f"saved train: {train_manifest}")
    print(f"saved valid: {valid_manifest}")


if __name__ == "__main__":
    main()
