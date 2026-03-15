import argparse
import subprocess
from pathlib import Path


def parse_args():
    # Keep the script generic so it can be reused for different codecs and datasets.
    parser = argparse.ArgumentParser(description="Batch-encode wav files and decode them back to degraded wavs.")
    parser.add_argument("--clean_dir", type=str, required=True, help="Root directory of clean wav files")
    parser.add_argument("--degraded_dir", type=str, required=True, help="Root directory of degraded wav outputs")
    parser.add_argument(
        "--codec",
        type=str,
        default="opus",
        choices=["opus", "g711_mulaw", "g711_alaw", "amrwb"],
        help="Codec to simulate",
    )
    parser.add_argument("--bitrate", type=str, default="8k", help="Target codec bitrate, e.g. 8k")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Output wav sample rate after decoding")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing degraded wav files")
    parser.add_argument(
        "--trials_file",
        type=str,
        default="",
        help="Optional trials file. When provided, only utterances used in the selected trials are degraded.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Only process the first fraction of items. With --trials_file, this applies to trials; otherwise to wav files.",
    )
    return parser.parse_args()


def run_ffmpeg(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_ffmpeg_commands(input_wav: Path, output_wav: Path, codec: str, bitrate: str, sample_rate: int):
    # Use a codec-specific intermediate file so the lossy encode/decode path is
    # explicit and easy to inspect when comparing communication codecs.
    if codec == "opus":
        temp_file = output_wav.with_suffix(".opus")
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-c:a", "libopus", "-b:a", bitrate, str(temp_file)
        ]
    elif codec == "g711_mulaw":
        temp_file = output_wav.with_name(output_wav.stem + ".mulaw.wav")
        # G.711 is a narrowband telephone codec, so encode at 8 kHz mono.
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", str(temp_file)
        ]
    elif codec == "g711_alaw":
        temp_file = output_wav.with_name(output_wav.stem + ".alaw.wav")
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-ar", "8000", "-ac", "1", "-c:a", "pcm_alaw", str(temp_file)
        ]
    elif codec == "amrwb":
        temp_file = output_wav.with_suffix(".amr")
        # AMR-WB is a 16 kHz wideband speech codec. The bitrate parameter should
        # be a valid AMR-WB rate such as 6.60k, 8.85k, 12.65k, or 23.85k.
        encode_cmd = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-ar", "16000", "-ac", "1", "-c:a", "libvo_amrwbenc", "-b:a", bitrate, "-f", "amr", str(temp_file)
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    # Decode back to wav because the downstream speaker model expects wav input.
    decode_cmd = [
        "ffmpeg", "-y", "-i", str(temp_file),
        "-ar", str(sample_rate), str(output_wav)
    ]
    return temp_file, encode_cmd, decode_cmd


def process_audio(input_wav: Path, output_wav: Path, codec: str, bitrate: str, sample_rate: int):
    temp_file, encode_cmd, decode_cmd = build_ffmpeg_commands(
        input_wav=input_wav,
        output_wav=output_wav,
        codec=codec,
        bitrate=bitrate,
        sample_rate=sample_rate,
    )

    run_ffmpeg(encode_cmd)
    run_ffmpeg(decode_cmd)
    temp_file.unlink(missing_ok=True)


def resolve_wav_from_utt(clean_dir: Path, utt_id: str) -> Path:
    # VCTK utterance ids are formatted as p225_001, so the speaker id is the
    # prefix before the first underscore and the file is speaker/utt.wav.
    speaker_id = utt_id.split("_", 1)[0]
    wav_path = clean_dir / speaker_id / f"{utt_id}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"Wav file not found for utterance {utt_id}: {wav_path}")
    return wav_path


def load_wavs_from_trials(clean_dir: Path, trials_file: Path, fraction: float):
    lines = [line.strip() for line in trials_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if fraction < 1.0:
        keep = max(1, int(len(lines) * fraction))
        lines = lines[:keep]

    utt_ids = []
    seen = set()
    for line in lines:
        utt1, utt2, _ = line.split()
        for utt in (utt1, utt2):
            if utt not in seen:
                seen.add(utt)
                utt_ids.append(utt)

    wav_files = [resolve_wav_from_utt(clean_dir, utt_id) for utt_id in utt_ids]
    return wav_files, len(lines)


def main():
    args = parse_args()
    clean_dir = Path(args.clean_dir).resolve()
    degraded_dir = Path(args.degraded_dir).resolve()

    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean wav directory not found: {clean_dir}")

    if not 0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in the range (0, 1].")

    if args.trials_file:
        trials_file = Path(args.trials_file).resolve()
        if not trials_file.exists():
            raise FileNotFoundError(f"Trials file not found: {trials_file}")
        # Degrade exactly the utterances referenced by the selected trials so
        # clean and degraded evaluation use the same underlying audio set.
        wav_files, num_trials = load_wavs_from_trials(clean_dir, trials_file, args.fraction)
        print(f"Selected {num_trials} trials and {len(wav_files)} referenced wav files")
    else:
        wav_files = sorted(clean_dir.rglob("*.wav"))
        if not wav_files:
            raise ValueError(f"No wav files found under {clean_dir}")
        if args.fraction < 1.0:
            keep = max(1, int(len(wav_files) * args.fraction))
            wav_files = wav_files[:keep]
        print(f"Found {len(wav_files)} wav files under {clean_dir}")

    print(f"Writing degraded wavs to {degraded_dir}")

    processed = 0
    skipped = 0
    for input_file in wav_files:
        rel_path = input_file.relative_to(clean_dir)
        output_file = degraded_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and not args.overwrite:
            skipped += 1
            continue

        print(f"Processing: {rel_path}")
        try:
            process_audio(input_file, output_file, args.codec, args.bitrate, args.sample_rate)
            processed += 1
        except Exception as exc:
            print(f"Failed: {rel_path} -> {exc}")

    print(f"Finished. Processed: {processed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
