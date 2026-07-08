#!/usr/bin/env python3
"""Prepare a CN-Celeb-style VoxCeleb1 evaluation set.

The script selects exactly one enrollment utterance per speaker, puts every
other unique utterance into the test set, and writes the Cartesian-product
trials.lst expected by the existing CN-Celeb evaluation pipeline.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath


DEFAULT_TRIALS = Path(
    "/root/autodl-tmp/raw_data/vox1/data/vox1_trials/veri_test2.txt"
)
DEFAULT_WAV_ROOT = Path("/root/autodl-tmp/raw_data/vox1/train/wav")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/SC_data/voxceleb_data_test_raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CN-Celeb-style evaluation set from VoxCeleb1 trials."
    )
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--wav-root", type=Path, default=DEFAULT_WAV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to place WAV files in the output tree (default: copy).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the expected result without writing files.",
    )
    return parser.parse_args()


def load_unique_utterances(trials_path: Path) -> dict[str, list[str]]:
    """Return sorted unique utterance paths grouped by speaker."""
    by_speaker: dict[str, set[str]] = defaultdict(set)
    with trials_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 3 or fields[0] not in {"0", "1"}:
                raise ValueError(
                    f"Malformed line {line_number} in {trials_path}: {line!r}"
                )
            for relative_path in fields[1:]:
                path = PurePosixPath(relative_path)
                if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                    raise ValueError(
                        f"Unsafe utterance path on line {line_number}: {relative_path!r}"
                    )
                by_speaker[path.parts[0]].add(relative_path)

    if not by_speaker:
        raise ValueError(f"No utterances found in {trials_path}")
    return {speaker: sorted(paths) for speaker, paths in sorted(by_speaker.items())}


def flat_name(relative_path: str) -> str:
    """Map id/video/file.wav to the collision-safe id-video-file.wav."""
    return "-".join(PurePosixPath(relative_path).parts)


def place_file(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        destination.symlink_to(source.resolve())


def main() -> int:
    args = parse_args()
    try:
        utterances = load_unique_utterances(args.trials)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    enroll: dict[str, str] = {}
    tests: list[tuple[str, str]] = []
    missing: list[Path] = []
    seen_names: set[str] = set()

    for speaker, paths in utterances.items():
        if len(paths) < 2:
            print(f"ERROR: speaker {speaker} has fewer than two utterances", file=sys.stderr)
            return 1
        enroll[speaker] = paths[0]
        for relative_path in paths:
            source = args.wav_root / relative_path
            if not source.is_file():
                missing.append(source)
        for relative_path in paths[1:]:
            name = flat_name(relative_path)
            if name in seen_names:
                print(f"ERROR: flattened filename collision: {name}", file=sys.stderr)
                return 1
            seen_names.add(name)
            tests.append((speaker, relative_path))

    if missing:
        print(f"ERROR: {len(missing)} source WAV files are missing.", file=sys.stderr)
        for path in missing[:10]:
            print(f"  {path}", file=sys.stderr)
        return 1

    speaker_count = len(enroll)
    test_count = len(tests)
    trial_count = speaker_count * test_count
    positive_count = test_count
    print(f"Speakers:       {speaker_count}")
    print(f"Enroll WAVs:    {speaker_count}")
    print(f"Test WAVs:      {test_count}")
    print(f"Trials:         {trial_count}")
    print(f"Positive:       {positive_count}")
    print(f"Negative:       {trial_count - positive_count}")
    print(f"Output:         {args.output_dir}")

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    if args.output_dir.exists():
        if not args.overwrite:
            print(
                f"ERROR: output directory already exists: {args.output_dir}\n"
                "Use --overwrite only if you want to replace it.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(args.output_dir)

    enroll_dir = args.output_dir / "enroll"
    test_dir = args.output_dir / "test"
    enroll_dir.mkdir(parents=True)
    test_dir.mkdir()

    for speaker, relative_path in enroll.items():
        source = args.wav_root / relative_path
        place_file(source, enroll_dir / f"{speaker}-enroll.wav", args.link_mode)

    for speaker, relative_path in tests:
        speaker_dir = test_dir / speaker
        speaker_dir.mkdir(exist_ok=True)
        place_file(source=args.wav_root / relative_path,
                   destination=speaker_dir / flat_name(relative_path),
                   mode=args.link_mode)

    trials_output = args.output_dir / "trials.lst"
    with trials_output.open("w", encoding="utf-8", newline="\n") as handle:
        for enroll_speaker in enroll:
            enroll_key = f"{enroll_speaker}-enroll"
            for test_speaker, relative_path in tests:
                label = 1 if enroll_speaker == test_speaker else 0
                test_key = f"test/{test_speaker}/{flat_name(relative_path)}"
                handle.write(f"{enroll_key} {test_key} {label}\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
