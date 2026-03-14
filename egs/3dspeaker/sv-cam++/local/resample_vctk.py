import argparse
import multiprocessing as mp
from pathlib import Path

import torchaudio


def parse_args():
    # Keep resampling as a standalone utility so the expensive conversion step
    # can be done once and reused across experiments.
    parser = argparse.ArgumentParser(description="Resample VCTK wavs to 16k.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Source VCTK wav directory, e.g. data/1/VCTK-Corpus/VCTK-Corpus/wav48",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Target directory for resampled wavs",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of worker processes",
    )
    return parser.parse_args()


def init_worker(target_sample_rate):
    # Store the target sample rate once per worker to avoid passing it in every task.
    global TARGET_SAMPLE_RATE
    TARGET_SAMPLE_RATE = target_sample_rate


def resample_one(task):
    # Mirror the original speaker/utterance directory structure under output_dir.
    src_path, dst_path = task
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        try:
            # Skip files that were already converted to the desired sample rate.
            info = torchaudio.info(str(dst_path))
            if info.sample_rate == TARGET_SAMPLE_RATE:
                return ("skip", str(dst_path))
        except Exception:
            pass

    wav, sample_rate = torchaudio.load(str(src_path))
    if wav.shape[0] > 1:
        # Downmix by keeping a single channel because the training pipeline
        # expects mono waveforms.
        wav = wav[:1, :]
    if sample_rate != TARGET_SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            wav, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
    torchaudio.save(str(dst_path), wav, TARGET_SAMPLE_RATE)
    return ("ok", str(dst_path))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    tasks = []
    for src_path in sorted(input_dir.rglob("*.wav")):
        # Preserve the relative path so p225/p225_001.wav stays under p225/.
        rel_path = src_path.relative_to(input_dir)
        dst_path = output_dir / rel_path
        tasks.append((str(src_path), str(dst_path)))

    if not tasks:
        raise ValueError(f"No wav files found under {input_dir}")

    print(f"Found {len(tasks)} wav files")
    print(f"Resampling to {args.sample_rate} Hz")
    print(f"Output directory: {output_dir}")

    ok_count = 0
    skip_count = 0
    with mp.Pool(args.num_workers, initializer=init_worker, initargs=(args.sample_rate,)) as pool:
        for status, path in pool.imap_unordered(resample_one, tasks, chunksize=16):
            if status == "ok":
                ok_count += 1
            else:
                skip_count += 1
            done = ok_count + skip_count
            if done % 500 == 0 or done == len(tasks):
                print(f"Processed {done}/{len(tasks)} files")

    print(f"Finished. Resampled: {ok_count}, skipped: {skip_count}")


if __name__ == "__main__":
    main()
