import argparse
import random
from pathlib import Path


def parse_args():
    # Expose split size controls so the same script can be used for debugging
    # on a small subset or for full-dataset preparation.
    parser = argparse.ArgumentParser(description="Prepare VCTK train/test split.")
    parser.add_argument("--wav_dir", type=str, required=True, help="Root dir of resampled VCTK wavs")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dir, e.g. data/vctk")
    parser.add_argument("--max_speakers", type=int, default=0, help="Use at most this many speakers, 0 means all")
    parser.add_argument(
        "--max_utts_per_speaker",
        type=int,
        default=0,
        help="Use at most this many utterances per speaker before split, 0 means all",
    )
    parser.add_argument("--num_train_utts", type=int, default=250, help="Train utterances per speaker")
    parser.add_argument("--num_test_utts", type=int, default=50, help="Test utterances per speaker")
    parser.add_argument("--num_target_trials", type=int, default=20, help="Target trials per speaker")
    parser.add_argument("--num_nontarget_trials", type=int, default=20, help="Nontarget trials per speaker")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser.parse_args()


def write_kaldi_files(split_dir: Path, rows):
    # Write the standard Kaldi metadata files expected by the downstream tools.
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_scp = split_dir / "wav.scp"
    utt2spk = split_dir / "utt2spk"
    spk2utt = split_dir / "spk2utt"

    rows = sorted(rows, key=lambda x: x[0])
    with wav_scp.open("w", encoding="utf-8") as fwav, utt2spk.open("w", encoding="utf-8") as futt:
        for utt, wav_path, spk in rows:
            fwav.write(f"{utt} {wav_path}\n")
            futt.write(f"{utt} {spk}\n")

    spk_map = {}
    for utt, _, spk in rows:
        spk_map.setdefault(spk, []).append(utt)
    with spk2utt.open("w", encoding="utf-8") as fspk:
        for spk in sorted(spk_map):
            utts = " ".join(sorted(spk_map[spk]))
            fspk.write(f"{spk} {utts}\n")


def make_trials(test_map, num_target_trials, num_nontarget_trials, rng):
    # Build a bounded number of positive and negative pairs per speaker to keep
    # scoring manageable on VCTK-sized experiments.
    speakers = sorted(test_map)
    lines = []
    for spk in speakers:
        utts = test_map[spk]
        target_pairs = []
        if len(utts) >= 2:
            for i in range(len(utts)):
                for j in range(i + 1, len(utts)):
                    target_pairs.append((utts[i], utts[j], "target"))
            rng.shuffle(target_pairs)
            lines.extend(target_pairs[: min(num_target_trials, len(target_pairs))])

        other_utts = []
        for other_spk in speakers:
            if other_spk != spk:
                other_utts.extend((u, other_spk) for u in test_map[other_spk])
        rng.shuffle(other_utts)
        for idx in range(min(num_nontarget_trials, len(utts), len(other_utts))):
            lines.append((utts[idx], other_utts[idx][0], "nontarget"))
    return lines


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    wav_dir = Path(args.wav_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not wav_dir.exists():
        raise FileNotFoundError(f"wav_dir not found: {wav_dir}")

    speaker_dirs = sorted([p for p in wav_dir.iterdir() if p.is_dir()])
    if args.max_speakers > 0:
        speaker_dirs = speaker_dirs[: args.max_speakers]

    train_rows = []
    test_rows = []
    test_map = {}

    for spk_dir in speaker_dirs:
        # Each speaker directory contains files like p225_001.wav.
        spk = spk_dir.name
        wavs = sorted(spk_dir.glob("*.wav"))
        if args.max_utts_per_speaker > 0:
            wavs = wavs[: args.max_utts_per_speaker]

        # Skip speakers that do not have enough utterances for the requested split.
        need = args.num_train_utts + args.num_test_utts
        if len(wavs) < need:
            continue

        # Use a deterministic slice so repeated runs with the same directory
        # layout yield the same split.
        train_wavs = wavs[: args.num_train_utts]
        test_wavs = wavs[args.num_train_utts : need]

        for wav in train_wavs:
            train_rows.append((wav.stem, str(wav), spk))
        for wav in test_wavs:
            test_rows.append((wav.stem, str(wav), spk))
        test_map[spk] = [wav.stem for wav in test_wavs]

    if not train_rows or not test_rows:
        raise ValueError("No valid train/test split generated. Check utterance counts per speaker.")

    write_kaldi_files(out_dir / "train", train_rows)
    write_kaldi_files(out_dir / "test", test_rows)

    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    trials = make_trials(test_map, args.num_target_trials, args.num_nontarget_trials, rng)
    with (trials_dir / "trials").open("w", encoding="utf-8") as f:
        for enrol, test, label in trials:
            f.write(f"{enrol} {test} {label}\n")

    print(f"Prepared {len(test_map)} speakers")
    print(f"Train utterances: {len(train_rows)}")
    print(f"Test utterances: {len(test_rows)}")
    print(f"Trials: {len(trials)}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
