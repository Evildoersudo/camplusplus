#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split pair manifest into train/valid by speaker id.")
    p.add_argument("--input_manifest", type=str, required=True)
    p.add_argument("--train_manifest", type=str, required=True)
    p.add_argument("--valid_manifest", type=str, required=True)
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    input_manifest = Path(args.input_manifest).resolve()
    rows = list(csv.DictReader(input_manifest.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"No rows found: {input_manifest}")

    spks = sorted({row["spk_id"] for row in rows})
    random.Random(args.seed).shuffle(spks)
    n_valid = max(1, int(len(spks) * args.valid_ratio))
    valid_spk = set(spks[:n_valid])

    train_rows = [row for row in rows if row["spk_id"] not in valid_spk]
    valid_rows = [row for row in rows if row["spk_id"] in valid_spk]

    fields = ["utt_id", "spk_id", "clean_wav", "codec_wav"]

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
    print(f"train rows={len(train_rows)}")
    print(f"valid rows={len(valid_rows)}")
    print(f"saved train: {train_manifest}")
    print(f"saved valid: {valid_manifest}")


if __name__ == "__main__":
    main()
