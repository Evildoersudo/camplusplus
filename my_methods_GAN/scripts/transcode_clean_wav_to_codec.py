#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import random
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcode existing clean wavs to codec/decode wavs for subset training.",
    )
    p.add_argument("--clean_root", type=str, required=True, help="Root dir of clean wavs (recursive).")
    p.add_argument("--coded_root", type=str, required=True, help="Output root dir for coded-decoded wavs.")
    p.add_argument(
        "--codec",
        type=str,
        required=True,
        choices=["opus", "aac", "amrwb", "g711_mulaw", "g711_alaw"],
        help="Codec used in encode->decode simulation.",
    )
    p.add_argument("--bitrate", type=str, default="16k", help="Codec bitrate, e.g. 16k, 12.65k.")
    p.add_argument("--sample_rate", type=int, default=16000, help="Decoded wav sample rate.")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers.")
    p.add_argument("--max_files", type=int, default=0, help="If >0, randomly sample this many wavs.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output wavs.")
    return p.parse_args()


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_ffmpeg_commands(input_wav: Path, output_wav: Path, codec: str, bitrate: str, sample_rate: int) -> tuple[Path, list[str], list[str]]:
    if codec == "opus":
        temp_file = output_wav.with_suffix(".opus")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-c:a",
            "libopus",
            "-b:a",
            bitrate,
            str(temp_file),
        ]
    elif codec == "aac":
        temp_file = output_wav.with_suffix(".m4a")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            str(temp_file),
        ]
    elif codec == "amrwb":
        temp_file = output_wav.with_suffix(".amr")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "libvo_amrwbenc",
            "-b:a",
            bitrate,
            "-f",
            "amr",
            str(temp_file),
        ]
    elif codec == "g711_mulaw":
        temp_file = output_wav.with_name(output_wav.stem + ".mulaw.wav")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "8000",
            "-ac",
            "1",
            "-c:a",
            "pcm_mulaw",
            str(temp_file),
        ]
    elif codec == "g711_alaw":
        temp_file = output_wav.with_name(output_wav.stem + ".alaw.wav")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "8000",
            "-ac",
            "1",
            "-c:a",
            "pcm_alaw",
            str(temp_file),
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    decode_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_file),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_wav),
    ]
    return temp_file, encode_cmd, decode_cmd


def transcode_one(clean_wav: Path, coded_wav: Path, codec: str, bitrate: str, sample_rate: int, overwrite: bool) -> None:
    if coded_wav.exists() and not overwrite:
        return
    coded_wav.parent.mkdir(parents=True, exist_ok=True)
    temp_file, encode_cmd, decode_cmd = build_ffmpeg_commands(clean_wav, coded_wav, codec, bitrate, sample_rate)
    run_ffmpeg(encode_cmd)
    run_ffmpeg(decode_cmd)
    temp_file.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    clean_root = Path(args.clean_root).resolve()
    coded_root = Path(args.coded_root).resolve()
    if not clean_root.exists():
        raise FileNotFoundError(f"clean_root not found: {clean_root}")

    clean_wavs = sorted(clean_root.rglob("*.wav"))
    if not clean_wavs:
        raise RuntimeError(f"No wav found under: {clean_root}")

    if args.max_files > 0 and args.max_files < len(clean_wavs):
        rng = random.Random(args.seed)
        clean_wavs = rng.sample(clean_wavs, args.max_files)
        clean_wavs.sort()

    print(f"clean_root: {clean_root}")
    print(f"coded_root: {coded_root}")
    print(f"codec={args.codec} bitrate={args.bitrate} sample_rate={args.sample_rate}")
    print(f"selected wavs: {len(clean_wavs)}")

    tasks: list[tuple[Path, Path, str, str, int, bool]] = []
    for clean_wav in clean_wavs:
        rel = clean_wav.relative_to(clean_root)
        coded_wav = coded_root / rel
        tasks.append((clean_wav, coded_wav, args.codec, args.bitrate, args.sample_rate, args.overwrite))

    total = len(tasks)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(transcode_one, *task) for task in tasks]
        for f in concurrent.futures.as_completed(futures):
            f.result()
            done += 1
            if done % 500 == 0 or done == total:
                print(f"transcode: {done}/{total}")

    print("done")


if __name__ == "__main__":
    main()
