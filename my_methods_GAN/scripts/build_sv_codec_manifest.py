#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sv_codec_restore_gan.data.manifest import build_pair_manifest


def parse_args():
    p = argparse.ArgumentParser(description="Build clean/coded pair manifest for SV-CodecRestoreGAN.")
    p.add_argument("--clean_root", type=str, required=True)
    p.add_argument(
        "--coded_root",
        type=str,
        required=True,
        nargs="+",
        help="One or more coded roots. Multiple roots are merged into multi-codec entries.",
    )
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument(
        "--include_manifest",
        "--include-manifest",
        type=str,
        default=None,
        help="Optional clean manifest selecting which utterances to include.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    n = build_pair_manifest(
        args.clean_root,
        args.coded_root,
        args.output_csv,
        include_manifest=args.include_manifest,
    )
    print(f"Saved manifest: {Path(args.output_csv).resolve()}")
    print(f"Paired utterances: {n}")


if __name__ == "__main__":
    main()
