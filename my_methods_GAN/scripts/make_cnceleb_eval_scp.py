#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build clean/coded wav.scp for CN-Celeb eval from trials list.")
    p.add_argument("--trials_file", type=str, required=True)
    p.add_argument("--eval_clean_dir", type=str, required=True, help="Directory containing enroll/ and test/ wavs.")
    p.add_argument("--eval_coded_dir", type=str, required=True, help="Directory containing enroll/ and test/ wavs.")
    p.add_argument("--clean_scp_out", type=str, default="", help="Output clean wav.scp path. Required unless --skip_clean_scp is set.")
    p.add_argument("--coded_scp_out", type=str, required=True)
    p.add_argument("--skip_clean_scp", action="store_true", help="Skip writing clean wav.scp; only write coded wav.scp.")
    p.add_argument("--strict", action="store_true", help="Fail if any trial utterance cannot be resolved.")
    return p.parse_args()


def _parse_trials(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            items.extend([parts[0], parts[1]])
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _resolve_key_to_path(utt_key: str, eval_dir: Path) -> Path:
    if utt_key.startswith("test/"):
        filename = utt_key.split("/", 1)[1]
        return (eval_dir / "test" / filename).resolve()

    if utt_key.endswith("-enroll"):
        return (eval_dir / "enroll" / f"{utt_key}.wav").resolve()

    if utt_key.endswith(".wav"):
        return (eval_dir / utt_key).resolve()

    return (eval_dir / f"{utt_key}.wav").resolve()


def _write_scp(keys: list[str], eval_dir: Path, out_path: Path, strict: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    with out_path.open("w", encoding="utf-8") as f:
        for key in keys:
            wav_path = _resolve_key_to_path(key, eval_dir)
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

    keys = _parse_trials(trials_file)
    print(f"trial utterances: {len(keys)}")

    if args.skip_clean_scp:
        print("skip clean scp generation due to --skip_clean_scp")
    else:
        _write_scp(keys, eval_clean_dir, clean_scp_out, args.strict)
    _write_scp(keys, eval_coded_dir, coded_scp_out, args.strict)

    if not args.skip_clean_scp:
        print(f"saved clean scp: {clean_scp_out}")
    print(f"saved coded scp: {coded_scp_out}")


if __name__ == "__main__":
    main()
