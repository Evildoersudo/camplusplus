import argparse
import csv
import random
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from simulate_codec import process_audio


def parse_args():
    # This script builds a mixed-condition training set from an existing
    # Kaldi-style training split. Each utterance is assigned to exactly one
    # condition so the output wav.scp stays unique and training can reuse the
    # existing prepare_data_csv.py workflow.
    parser = argparse.ArgumentParser(
        description="Build a mixed clean/codec training set and generate wav.scp, utt2spk, spk2utt, and train.csv."
    )
    parser.add_argument("--input_wav_scp", type=str, required=True, help="Source wav.scp")
    parser.add_argument("--input_utt2spk", type=str, required=True, help="Source utt2spk")
    parser.add_argument("--output_dir", type=str, required=True, help="Output Kaldi-style data directory")
    parser.add_argument(
        "--audio_output_root",
        type=str,
        required=True,
        help="Root directory where mixed audio files will be written",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Decoded wav sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel ffmpeg worker count")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing degraded wavs")
    parser.add_argument(
        "--max_utts",
        type=int,
        default=0,
        help="Optional cap for debugging. 0 means use all utterances.",
    )
    parser.add_argument(
        "--clean_ratio",
        type=float,
        default=0.30,
        help="Fraction kept as clean audio",
    )
    parser.add_argument(
        "--opus_ratio",
        type=float,
        default=0.30,
        help="Fraction encoded with Opus",
    )
    parser.add_argument(
        "--amrwb_ratio",
        type=float,
        default=0.20,
        help="Fraction encoded with AMR-WB",
    )
    parser.add_argument(
        "--g711_ratio",
        type=float,
        default=0.20,
        help="Fraction encoded with G.711",
    )
    parser.add_argument(
        "--opus_bitrates",
        type=str,
        default="4k,6k,8k",
        help="Comma-separated Opus bitrate candidates",
    )
    parser.add_argument(
        "--amrwb_bitrates",
        type=str,
        default="8.85k,8.85k,8.85k,12.65k,23.85k",
        help="Comma-separated AMR-WB bitrate candidates. Repeating 8.85k increases its sampling probability.",
    )
    parser.add_argument(
        "--g711_variants",
        type=str,
        default="g711_mulaw,g711_alaw",
        help="Comma-separated G.711 codec variants",
    )
    parser.add_argument(
        "--skip_train_csv",
        action="store_true",
        help="Skip calling prepare_data_csv.py",
    )
    parser.add_argument(
        "--prepare_csv_nj",
        type=int,
        default=8,
        help="Process count passed to prepare_data_csv.py",
    )
    return parser.parse_args()


def load_kaldi_map(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            mapping[key] = value
    return mapping


def write_kaldi_files(output_dir: Path, wav_scp, utt2spk):
    wav_scp_path = output_dir / "wav.scp"
    utt2spk_path = output_dir / "utt2spk"
    spk2utt_path = output_dir / "spk2utt"

    spk_to_utts = defaultdict(list)
    for utt_id, spk_id in utt2spk.items():
        spk_to_utts[spk_id].append(utt_id)

    with wav_scp_path.open("w", encoding="utf-8", newline="\n") as fwav:
        for utt_id in sorted(wav_scp):
            fwav.write(f"{utt_id} {wav_scp[utt_id]}\n")

    with utt2spk_path.open("w", encoding="utf-8", newline="\n") as futt:
        for utt_id in sorted(utt2spk):
            futt.write(f"{utt_id} {utt2spk[utt_id]}\n")

    with spk2utt_path.open("w", encoding="utf-8", newline="\n") as fspk:
        for spk_id in sorted(spk_to_utts):
            utts = " ".join(sorted(spk_to_utts[spk_id]))
            fspk.write(f"{spk_id} {utts}\n")


def choose_condition(index, counts):
    # Assign contiguous shuffled segments to each condition so ratios match the
    # requested counts exactly after rounding.
    if index < counts["clean"]:
        return "clean"
    index -= counts["clean"]
    if index < counts["opus"]:
        return "opus"
    index -= counts["opus"]
    if index < counts["amrwb"]:
        return "amrwb"
    return "g711"


def build_assignment(entries, args):
    total = len(entries)
    clean_count = int(total * args.clean_ratio)
    opus_count = int(total * args.opus_ratio)
    amrwb_count = int(total * args.amrwb_ratio)
    g711_count = total - clean_count - opus_count - amrwb_count

    counts = {
        "clean": clean_count,
        "opus": opus_count,
        "amrwb": amrwb_count,
        "g711": g711_count,
    }

    opus_bitrates = [item.strip() for item in args.opus_bitrates.split(",") if item.strip()]
    amrwb_bitrates = [item.strip() for item in args.amrwb_bitrates.split(",") if item.strip()]
    g711_variants = [item.strip() for item in args.g711_variants.split(",") if item.strip()]

    assignments = []
    for index, (utt_id, wav_path, spk_id) in enumerate(entries):
        condition = choose_condition(index, counts)
        codec = "clean"
        bitrate = "-"

        if condition == "opus":
            codec = "opus"
            bitrate = random.choice(opus_bitrates)
        elif condition == "amrwb":
            codec = "amrwb"
            bitrate = random.choice(amrwb_bitrates)
        elif condition == "g711":
            codec = random.choice(g711_variants)
            bitrate = "8k"

        assignments.append(
            {
                "utt_id": utt_id,
                "spk_id": spk_id,
                "input_wav": Path(wav_path).resolve(),
                "condition": condition,
                "codec": codec,
                "bitrate": bitrate,
            }
        )

    return assignments


def output_wav_path(audio_root: Path, item):
    if item["condition"] == "clean":
        return item["input_wav"]

    # Keep a codec-specific directory layout so the resulting corpus can be
    # inspected and reused later without regenerating audio.
    codec_tag = item["codec"]
    bitrate_tag = item["bitrate"].replace(".", "").replace("k", "k")
    if item["bitrate"] == "-":
        bitrate_tag = "default"
    speaker_dir = item["spk_id"]
    file_name = f"{item['utt_id']}.wav"
    return audio_root / codec_tag / bitrate_tag / speaker_dir / file_name


def process_assignment(item, audio_root: Path, sample_rate: int, overwrite: bool):
    output_wav = output_wav_path(audio_root, item)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    if item["condition"] == "clean":
        return str(item["input_wav"]), "copied-clean"

    if output_wav.exists() and not overwrite:
        return str(output_wav), "reused"

    process_audio(
        input_wav=item["input_wav"],
        output_wav=output_wav,
        codec=item["codec"],
        bitrate=item["bitrate"],
        sample_rate=sample_rate,
    )
    return str(output_wav), "generated"


def write_manifest(output_dir: Path, assignments, wav_map):
    manifest_path = output_dir / "codec_assignment.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utt_id", "spk_id", "condition", "codec", "bitrate", "path"])
        for item in assignments:
            writer.writerow(
                [
                    item["utt_id"],
                    item["spk_id"],
                    item["condition"],
                    item["codec"],
                    item["bitrate"],
                    wav_map[item["utt_id"]],
                ]
            )


def maybe_prepare_train_csv(output_dir: Path, sample_rate: int, nj: int):
    script_path = Path(__file__).resolve().parent / "prepare_data_csv.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--data_dir",
        str(output_dir),
        "--sample_rate",
        str(sample_rate),
        "--nj",
        str(nj),
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    random.seed(args.seed)

    total_ratio = args.clean_ratio + args.opus_ratio + args.amrwb_ratio + args.g711_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("clean_ratio + opus_ratio + amrwb_ratio + g711_ratio must equal 1.0")

    input_wav_scp = Path(args.input_wav_scp).resolve()
    input_utt2spk = Path(args.input_utt2spk).resolve()
    output_dir = Path(args.output_dir).resolve()
    audio_output_root = Path(args.audio_output_root).resolve()

    wav_scp = load_kaldi_map(input_wav_scp)
    utt2spk = load_kaldi_map(input_utt2spk)

    common_utts = sorted(set(wav_scp).intersection(utt2spk))
    if not common_utts:
        raise ValueError("No overlapping utterances between wav.scp and utt2spk")

    entries = [(utt_id, wav_scp[utt_id], utt2spk[utt_id]) for utt_id in common_utts]
    random.shuffle(entries)
    if args.max_utts > 0:
        entries = entries[: args.max_utts]

    assignments = build_assignment(entries, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_root.mkdir(parents=True, exist_ok=True)

    wav_out = {}
    utt2spk_out = {}
    stats = defaultdict(int)

    # FFmpeg work is external process work, so threads are enough here and
    # simpler than process-based orchestration.
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_map = {
            executor.submit(
                process_assignment,
                item,
                audio_output_root,
                args.sample_rate,
                args.overwrite,
            ): item
            for item in assignments
        }

        for future in as_completed(future_map):
            item = future_map[future]
            output_path, status = future.result()
            wav_out[item["utt_id"]] = output_path
            utt2spk_out[item["utt_id"]] = item["spk_id"]
            stats[item["condition"]] += 1

    write_kaldi_files(output_dir, wav_out, utt2spk_out)
    write_manifest(output_dir, assignments, wav_out)

    if not args.skip_train_csv:
        maybe_prepare_train_csv(output_dir, args.sample_rate, args.prepare_csv_nj)

    print("Mixed codec dataset preparation finished")
    print(f"Output data dir: {output_dir}")
    print(f"Output audio root: {audio_output_root}")
    print(f"Total utterances: {len(assignments)}")
    print(f"Clean: {stats['clean']}")
    print(f"Opus: {stats['opus']}")
    print(f"AMR-WB: {stats['amrwb']}")
    print(f"G.711: {stats['g711']}")


if __name__ == "__main__":
    main()
