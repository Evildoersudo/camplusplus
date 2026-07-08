#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Build clean/coded wav.scp for CN-Celeb or VoxCeleb verification trials."
    )
    p.add_argument("--trials_file", type=str, required=True)
    p.add_argument("--eval_clean_dir", type=str, required=True, help="Directory containing enroll/ and test/ wavs.")
    p.add_argument("--eval_coded_dir", type=str, required=True, help="Directory containing enroll/ and test/ wavs.")
    p.add_argument("--clean_scp_out", type=str, default="", help="Output clean wav.scp path. Required unless --skip_clean_scp is set.")
    p.add_argument("--coded_scp_out", type=str, required=True)
    p.add_argument("--skip_clean_scp", action="store_true", help="Skip writing clean wav.scp; only write coded wav.scp.")
    p.add_argument("--strict", action="store_true", help="Fail if any trial utterance cannot be resolved.")
    return p.parse_args()


def _parse_label(value: str) -> int:
    value = value.strip().lower()
    if value in {"1", "target", "true"}:
        return 1
    if value in {"0", "nontarget", "non-target", "false"}:
        return 0
    raise ValueError(f"Unsupported trial label: {value}")


def _parse_trials(path: Path):
    """Read either ``utt1 utt2 label`` or ``label utt1 utt2`` trials."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed trial line {line_number} in {path}: {line.strip()!r}"
                )
            try:
                label = _parse_label(parts[2])
                utt1, utt2 = parts[0], parts[1]
            except ValueError:
                label = _parse_label(parts[0])
                utt1, utt2 = parts[1], parts[2]
            rows.append((label, utt1, utt2))

    items = [utt for _, utt1, utt2 in rows for utt in (utt1, utt2)]
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return rows, out


def _vox_flat_name(utt_key: str) -> str:
    return "-".join(Path(utt_key).parts)


def _build_legacy_enroll_map(keys: list[str]) -> dict[str, str]:
    """Recover the utterance selected by prepare_vox1_test_raw.py as enroll."""
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        parts = Path(key).parts
        if len(parts) >= 2 and parts[0].startswith("id") and key.endswith(".wav"):
            by_speaker[parts[0]].append(key)
    return {
        speaker: sorted(speaker_keys)[0]
        for speaker, speaker_keys in by_speaker.items()
    }


def _resolve_key_to_path(
    utt_key: str,
    eval_dir: Path,
    legacy_enroll_map: dict[str, str],
) -> Path:
    # Native VoxCeleb layout: <root>/id/video/file.wav.
    direct = (eval_dir / utt_key).resolve()
    if direct.is_file():
        return direct

    if utt_key.startswith("test/"):
        filename = utt_key.split("/", 1)[1]
        return (eval_dir / "test" / filename).resolve()

    if utt_key.endswith("-enroll"):
        return (eval_dir / "enroll" / f"{utt_key}.wav").resolve()

    # Compatibility with the old generated VoxCeleb tree:
    # test/id10270/id10270-video-00001.wav, with one utterance per speaker
    # stored as enroll/id10270-enroll.wav.
    parts = Path(utt_key).parts
    if len(parts) >= 2 and parts[0].startswith("id") and utt_key.endswith(".wav"):
        speaker = parts[0]
        flattened = (
            eval_dir / "test" / speaker / _vox_flat_name(utt_key)
        ).resolve()
        if flattened.is_file():
            return flattened
        if legacy_enroll_map.get(speaker) == utt_key:
            legacy_enroll = (eval_dir / "enroll" / f"{speaker}-enroll.wav").resolve()
            if legacy_enroll.is_file():
                return legacy_enroll

    if utt_key.endswith(".wav"):
        return (eval_dir / utt_key).resolve()

    return (eval_dir / f"{utt_key}.wav").resolve()


def _write_scp(
    keys: list[str],
    eval_dir: Path,
    out_path: Path,
    strict: bool,
    legacy_enroll_map: dict[str, str],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    with out_path.open("w", encoding="utf-8") as f:
        for key in keys:
            wav_path = _resolve_key_to_path(key, eval_dir, legacy_enroll_map)
            if not wav_path.exists():
                missing.append((key, wav_path))
                continue
            f.write(f"{key} {wav_path}\n")

    if missing:
        print(f"missing count: {len(missing)}")
        for key, wav_path in missing[:20]:
            print(f"MISSING {key} -> {wav_path}")
        if strict:
            raise FileNotFoundError(f"Missing {len(missing)} utterances for {out_path}")


def main():
    args = parse_args()
    trials_file = Path(args.trials_file).resolve()
    eval_clean_dir = Path(args.eval_clean_dir).resolve()
    eval_coded_dir = Path(args.eval_coded_dir).resolve()
    clean_scp_out = Path(args.clean_scp_out).resolve() if args.clean_scp_out else None
    coded_scp_out = Path(args.coded_scp_out).resolve()

    if not args.skip_clean_scp and clean_scp_out is None:
        raise ValueError("--clean_scp_out is required unless --skip_clean_scp is set")

    rows, keys = _parse_trials(trials_file)
    legacy_enroll_map = _build_legacy_enroll_map(keys)
    positive = sum(label == 1 for label, _, _ in rows)
    print(f"trials: {len(rows)} (positive={positive}, negative={len(rows) - positive})")
    print(f"unique trial utterances: {len(keys)}")

    if args.skip_clean_scp:
        print("skip clean scp generation due to --skip_clean_scp")
    else:
        _write_scp(
            keys, eval_clean_dir, clean_scp_out, args.strict, legacy_enroll_map
        )
    _write_scp(
        keys, eval_coded_dir, coded_scp_out, args.strict, legacy_enroll_map
    )

    if not args.skip_clean_scp:
        print(f"saved clean scp: {clean_scp_out}")
    print(f"saved coded scp: {coded_scp_out}")


if __name__ == "__main__":
    main()
