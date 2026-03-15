import argparse
import json
from pathlib import Path

from modelscope.pipelines import pipeline

from evaluate_eer import evaluate_trials, load_trials
from simulate_codec import load_wavs_from_trials, process_audio


def parse_args():
    # This script glues together degraded-audio generation and pretrained-model
    # evaluation so the full baseline can be reproduced with one command.
    parser = argparse.ArgumentParser(
        description="Generate degraded wavs referenced by trials and evaluate clean/degraded EER with a pretrained model."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="iic/speech_campplus_sv_zh-cn_16k-common",
        help="ModelScope model id",
    )
    parser.add_argument("--trials_file", type=str, required=True, help="Trials file")
    parser.add_argument("--clean_wav_root", type=str, required=True, help="Root directory of clean wavs")
    parser.add_argument("--degraded_wav_root", type=str, required=True, help="Root directory of degraded wavs")
    parser.add_argument(
        "--bitrate",
        type=str,
        default="8k",
        help="Target codec bitrate, e.g. 8k for Opus or 12.65k for AMR-WB",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Output wav sample rate after decoding")
    parser.add_argument(
        "--codec",
        type=str,
        default="opus",
        choices=["opus", "g711_mulaw", "g711_alaw", "amrwb"],
        help="Codec to simulate",
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="Only use the first fraction of trials")
    parser.add_argument("--limit", type=int, default=0, help="Only use the first N trials after fractioning")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing degraded wav files")
    parser.add_argument(
        "--report_file",
        type=str,
        default="",
        help="Optional path to save a JSON summary report",
    )
    return parser.parse_args()


def build_degraded_subset(clean_root: Path, degraded_root: Path, trials_file: Path, fraction: float, bitrate: str, sample_rate: int, codec: str, overwrite: bool):
    # Only degrade utterances that actually appear in the selected trials so
    # clean and degraded evaluation always operate on the same underlying set.
    wav_files, num_trials = load_wavs_from_trials(clean_root, trials_file, fraction)
    print(f"Selected {num_trials} trials and {len(wav_files)} referenced wav files for degradation")

    processed = 0
    skipped = 0
    for input_file in wav_files:
        rel_path = input_file.relative_to(clean_root)
        output_file = degraded_root / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and not overwrite:
            skipped += 1
            continue

        print(f"Degrading: {rel_path}")
        process_audio(input_file, output_file, codec, bitrate, sample_rate)
        processed += 1

    print(f"Degraded audio generation finished. Processed: {processed}, skipped: {skipped}")


def main():
    args = parse_args()
    trials_file = Path(args.trials_file).resolve()
    clean_wav_root = Path(args.clean_wav_root).resolve()
    degraded_wav_root = Path(args.degraded_wav_root).resolve()

    if not trials_file.exists():
        raise FileNotFoundError(f"Trials file not found: {trials_file}")
    if not clean_wav_root.exists():
        raise FileNotFoundError(f"Clean wav root not found: {clean_wav_root}")
    if not 0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in the range (0, 1].")

    build_degraded_subset(
        clean_root=clean_wav_root,
        degraded_root=degraded_wav_root,
        trials_file=trials_file,
        fraction=args.fraction,
        bitrate=args.bitrate,
        sample_rate=args.sample_rate,
        codec=args.codec,
        overwrite=args.overwrite,
    )

    print("Loading pretrained speaker verification pipeline...")
    sv_pipeline = pipeline(task="speaker-verification", model=args.model_id)

    trials = load_trials(trials_file, args.limit, args.fraction)
    clean_eer = evaluate_trials(sv_pipeline, trials, clean_wav_root, tag="Clean")
    degraded_eer = evaluate_trials(sv_pipeline, trials, degraded_wav_root, tag="Degraded")

    result = {
        "model_id": args.model_id,
        "trials_file": str(trials_file),
        "clean_wav_root": str(clean_wav_root),
        "degraded_wav_root": str(degraded_wav_root),
        "codec": args.codec,
        "bitrate": args.bitrate,
        "sample_rate": args.sample_rate,
        "fraction": args.fraction,
        "limit": args.limit,
        "clean_eer": clean_eer,
        "degraded_eer": degraded_eer,
        "degradation_factor": (degraded_eer / clean_eer) if clean_eer > 0 else None,
    }

    print("Summary")
    print(f"Clean EER: {clean_eer:.3f}%")
    print(f"Degraded EER: {degraded_eer:.3f}%")
    if clean_eer > 0:
        print(f"EER degradation factor: {degraded_eer / clean_eer:.3f}x")
    else:
        print("EER degradation factor: undefined because clean EER is 0")

    if args.report_file:
        report_file = Path(args.report_file).resolve()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved report to {report_file}")


if __name__ == "__main__":
    main()
