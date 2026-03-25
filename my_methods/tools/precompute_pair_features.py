"""Precompute CA-AFC training features into `.pt` files.

This script converts `(clean_wav, codec_wav)` rows from a pair manifest into
aligned tensors:
- clean_feat
- codec_feat
- aux_feat

The output is a feature manifest CSV consumed by `train_ca_afc.py`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_methods.data.ca_afc_data import extract_pair_features, read_pair_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CA-AFC pair features into .pt files.")
    parser.add_argument("--pair_manifest", type=str, required=True, help="Input pair manifest CSV from build_pair_manifest.py")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory used to store .pt feature files")
    parser.add_argument("--output_manifest", type=str, required=True, help="Output CSV listing all generated feature_pt files")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate used during feature extraction")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate .pt feature files even if they already exist")
    return parser.parse_args()


def feature_output_path(output_root: Path, row) -> Path:
    """Map one utterance to its precomputed feature file path."""

    return output_root / row.codec / row.bitrate.replace(".", "").replace("/", "_") / row.spk_id / f"{row.utt_id}.pt"


def main():
    args = parse_args()
    pair_manifest = Path(args.pair_manifest).resolve()
    output_root = Path(args.output_root).resolve()
    output_manifest = Path(args.output_manifest).resolve()

    rows = read_pair_manifest(pair_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    reused = 0
    manifest_rows = []
    for idx, row in enumerate(rows, start=1):
        feature_pt = feature_output_path(output_root, row)
        feature_pt.parent.mkdir(parents=True, exist_ok=True)

        if feature_pt.exists() and not args.overwrite:
            payload = torch.load(str(feature_pt), map_location="cpu")
            reused += 1
        else:
            payload = extract_pair_features(row.clean_wav, row.codec_wav, sample_rate=args.sample_rate)
            torch.save(payload, str(feature_pt))
            written += 1

        manifest_rows.append(
            {
                "utt_id": row.utt_id,
                "spk_id": row.spk_id,
                "feature_pt": str(feature_pt),
                "length": int(payload["length"]),
                "condition": row.condition,
                "codec": row.codec,
                "bitrate": row.bitrate,
            }
        )

        if idx % 200 == 0 or idx == len(rows):
            print(f"feature progress: {idx}/{len(rows)} (written={written}, reused={reused})")

    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["utt_id", "spk_id", "feature_pt", "length", "condition", "codec", "bitrate"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved feature manifest: {output_manifest}")
    print(f"Total rows: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
