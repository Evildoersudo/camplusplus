"""Build paired `(clean_wav, codec_wav)` manifests for CA-AFC training.

Typical use:
- `clean_wav.scp` comes from CN-Celeb clean training split.
- `mixed wav.scp` and `codec_assignment.csv` come from the codec-mixed dataset
  generated earlier by `build_mixed_codec_trainset.py`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_kaldi_map(path: Path):
    """Read a standard Kaldi two-column mapping file."""

    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            mapping[key] = value
    return mapping


def load_assignment(path: Path):
    """Load condition/codec metadata if `codec_assignment.csv` is available."""

    meta = {}
    if not path or not path.exists():
        return meta
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            meta[row["utt_id"]] = row
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description="Build CA-AFC pair manifest from clean and codec wav lists.")
    parser.add_argument("--clean_wav_scp", type=str, required=True, help="Clean wav.scp with original wav paths")
    parser.add_argument("--codec_wav_scp", type=str, required=True, help="Codec or mixed wav.scp used as CA-AFC input")
    parser.add_argument("--utt2spk", type=str, required=True, help="utt2spk for speaker ids")
    parser.add_argument("--codec_assignment_csv", type=str, default="", help="Optional codec_assignment.csv with codec metadata")
    parser.add_argument("--output_csv", type=str, required=True, help="Output pair manifest CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    clean_map = load_kaldi_map(Path(args.clean_wav_scp).resolve())
    codec_map = load_kaldi_map(Path(args.codec_wav_scp).resolve())
    utt2spk = load_kaldi_map(Path(args.utt2spk).resolve())
    assignment = load_assignment(Path(args.codec_assignment_csv).resolve()) if args.codec_assignment_csv else {}

    utts = sorted(set(clean_map).intersection(codec_map).intersection(utt2spk))
    if not utts:
        raise ValueError("No overlapping utterances among clean_wav_scp, codec_wav_scp, and utt2spk.")

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utt_id", "spk_id", "clean_wav", "codec_wav", "condition", "codec", "bitrate"])
        for utt_id in utts:
            meta = assignment.get(utt_id, {})
            writer.writerow(
                [
                    utt_id,
                    utt2spk[utt_id],
                    Path(clean_map[utt_id]).resolve(),
                    Path(codec_map[utt_id]).resolve(),
                    meta.get("condition", "unknown"),
                    meta.get("codec", "unknown"),
                    meta.get("bitrate", "-"),
                ]
            )

    print(f"Saved pair manifest: {output_csv}")
    print(f"Utterance count: {len(utts)}")


if __name__ == "__main__":
    main()
