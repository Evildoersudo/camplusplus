#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Build true-paired CN-Celeb clean/coded datasets from raw flac.",
    )
    p.add_argument(
        "--raw_root",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac",
        help="Root of CN-Celeb_flac containing data/dev/eval.",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="my_methods_GAN/data/cnceleb_truepair",
        help="Output root for clean/coded paired datasets.",
    )
    p.add_argument(
        "--codec",
        type=str,
        default="opus",
        choices=["opus", "aac", "amrwb", "g711_mulaw", "g711_alaw"],
        help="Codec used to generate coded counterparts.",
    )
    p.add_argument("--bitrate", type=str, default="16k", help="Codec bitrate, e.g. 16k, 12.65k.")
    p.add_argument("--sample_rate", type=int, default=16000, help="Decoded wav sample rate.")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--skip_train", action="store_true", help="Skip generating training split.")
    p.add_argument("--skip_eval", action="store_true", help="Skip generating eval split.")
    return p.parse_args()


def run_ffmpeg(cmd: list[str]):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcode_flac_to_wav(src: Path, dst: Path, sample_rate: int, overwrite: bool):
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(dst),
    ]
    run_ffmpeg(cmd)


def build_ffmpeg_commands(input_wav: Path, output_wav: Path, codec: str, bitrate: str, sample_rate: int):
    if codec == "opus":
        temp_file = output_wav.with_suffix(".opus")
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-c:a", "libopus", "-b:a", bitrate, str(temp_file),
        ]
    elif codec == "aac":
        temp_file = output_wav.with_suffix(".m4a")
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-ar", "16000", "-ac", "1", "-c:a", "aac", "-b:a", bitrate, str(temp_file),
        ]
    elif codec == "amrwb":
        temp_file = output_wav.with_suffix(".amr")
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-ar", "16000", "-ac", "1", "-c:a", "libvo_amrwbenc", "-b:a", bitrate, "-f", "amr", str(temp_file),
        ]
    elif codec == "g711_mulaw":
        temp_file = output_wav.with_name(output_wav.stem + ".mulaw.wav")
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(temp_file),
        ]
    elif codec == "g711_alaw":
        temp_file = output_wav.with_name(output_wav.stem + ".alaw.wav")
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw", str(temp_file),
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    decode_cmd = [
        "ffmpeg", "-y", "-i", str(temp_file),
        "-ar", str(sample_rate), "-ac", "1", str(output_wav),
    ]
    return temp_file, encode_cmd, decode_cmd


def transcode_wav_to_coded(input_wav: Path, output_wav: Path, codec: str, bitrate: str, sample_rate: int, overwrite: bool):
    if output_wav.exists() and not overwrite:
        return
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    temp_file, encode_cmd, decode_cmd = build_ffmpeg_commands(input_wav, output_wav, codec, bitrate, sample_rate)
    run_ffmpeg(encode_cmd)
    run_ffmpeg(decode_cmd)
    temp_file.unlink(missing_ok=True)


def _run_parallel(tasks, fn, workers: int, desc: str):
    if not tasks:
        print(f"{desc}: no tasks")
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(fn, *task) for task in tasks]
        done = 0
        total = len(futures)
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done += 1
            if done % 1000 == 0 or done == total:
                print(f"{desc}: {done}/{total}")


def main():
    args = parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.output_root).resolve()
    bitrate_tag = args.bitrate.replace(".", "")
    coded_train_dir = out_root / f"coded_train_{args.codec}_{bitrate_tag}"
    coded_eval_dir = out_root / f"eval_coded_{args.codec}_{bitrate_tag}"
    clean_train_dir = out_root / "clean_train_wav"
    eval_clean_dir = out_root / "eval_clean"

    dev_lst = raw_root / "dev" / "dev.lst"
    if not dev_lst.exists():
        raise FileNotFoundError(f"Missing dev list: {dev_lst}")

    dev_spk = {line.strip() for line in dev_lst.read_text(encoding="utf-8").splitlines() if line.strip()}

    if not args.skip_train:
        train_flacs = [
            path for path in (raw_root / "data").rglob("*.flac")
            if path.parent.name in dev_spk
        ]
        print(f"train flac count: {len(train_flacs)}")

        clean_tasks = []
        for src in train_flacs:
            rel = src.relative_to(raw_root / "data").with_suffix(".wav")
            dst = clean_train_dir / rel
            clean_tasks.append((src, dst, args.sample_rate, args.overwrite))

        _run_parallel(clean_tasks, transcode_flac_to_wav, args.workers, "train clean")

        coded_tasks = []
        for src in train_flacs:
            rel = src.relative_to(raw_root / "data").with_suffix(".wav")
            clean_wav = clean_train_dir / rel
            coded_wav = coded_train_dir / rel
            coded_tasks.append((clean_wav, coded_wav, args.codec, args.bitrate, args.sample_rate, args.overwrite))

        _run_parallel(coded_tasks, transcode_wav_to_coded, args.workers, "train coded")

    if not args.skip_eval:
        eval_flacs = list((raw_root / "eval" / "enroll").glob("*.flac")) + list((raw_root / "eval" / "test").glob("*.flac"))
        print(f"eval flac count: {len(eval_flacs)}")

        eval_clean_tasks = []
        for src in eval_flacs:
            part = src.parent.name
            dst = (eval_clean_dir / part / src.name).with_suffix(".wav")
            eval_clean_tasks.append((src, dst, args.sample_rate, args.overwrite))

        _run_parallel(eval_clean_tasks, transcode_flac_to_wav, args.workers, "eval clean")

        eval_coded_tasks = []
        for src in eval_flacs:
            part = src.parent.name
            clean_wav = (eval_clean_dir / part / src.name).with_suffix(".wav")
            coded_wav = (coded_eval_dir / part / src.name).with_suffix(".wav")
            eval_coded_tasks.append((clean_wav, coded_wav, args.codec, args.bitrate, args.sample_rate, args.overwrite))

        _run_parallel(eval_coded_tasks, transcode_wav_to_coded, args.workers, "eval coded")

    print("done")
    print(f"clean_train: {clean_train_dir}")
    print(f"coded_train: {coded_train_dir}")
    print(f"eval_clean:  {eval_clean_dir}")
    print(f"eval_coded:  {coded_eval_dir}")


if __name__ == "__main__":
    main()
