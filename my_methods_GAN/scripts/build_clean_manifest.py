#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean-only manifest from an id/video/*.wav tree."
    )
    parser.add_argument("--clean_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_root = args.clean_root.resolve()
    output_csv = args.output_csv.resolve()

    if not clean_root.is_dir():
        raise FileNotFoundError(f"Missing clean WAV root: {clean_root}")

    wavs = sorted(clean_root.glob("*/*/*.wav"))
    if not wavs:
        raise RuntimeError(f"No id/video/*.wav files found under {clean_root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utt_id", "spk_id", "clean_wav"])
        for wav in wavs:
            rel = wav.relative_to(clean_root)
            spk_id = rel.parts[0]
            utt_id = rel.with_suffix("").as_posix().replace("/", "-")
            writer.writerow([utt_id, spk_id, str(wav)])

    print(f"Saved manifest: {output_csv}")
    print(f"Clean utterances: {len(wavs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
