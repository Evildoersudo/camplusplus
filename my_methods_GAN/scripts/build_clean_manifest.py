#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean-only manifest from an id/video/audio tree."
    )
    parser.add_argument("--clean_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument(
        "--extensions",
        default=".wav",
        help="Comma-separated audio extensions to scan. Example: .m4a,.wav",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_root = args.clean_root.resolve()
    output_csv = args.output_csv.resolve()

    if not clean_root.is_dir():
        raise FileNotFoundError(f"Missing clean audio root: {clean_root}")

    extensions = {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.extensions.split(",")
        if ext.strip()
    }
    if not extensions:
        raise ValueError("--extensions must contain at least one extension")

    audio_files = sorted(
        path
        for path in clean_root.glob("*/*/*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not audio_files:
        raise RuntimeError(
            f"No id/video/audio files with extensions {sorted(extensions)} found under {clean_root}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utt_id", "spk_id", "video_id", "rel_path", "clean_wav"])
        for audio_path in audio_files:
            rel = audio_path.relative_to(clean_root)
            spk_id = rel.parts[0]
            video_id = rel.parts[1]
            utt_id = rel.with_suffix("").as_posix().replace("/", "-")
            writer.writerow([utt_id, spk_id, video_id, rel.as_posix(), str(audio_path)])

    print(f"Saved manifest: {output_csv}")
    print(f"Clean utterances: {len(audio_files)}")
    print(f"Extensions: {','.join(sorted(extensions))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
