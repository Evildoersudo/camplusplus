#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.compliance.kaldi as Kaldi

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sv_codec_restore_gan.models.campplus_wrapper import FrozenCampPlus
from sv_codec_restore_gan.utils.audio import load_audio_mono
from sv_codec_restore_gan.utils.metrics import compute_eer_mindcf


def parse_label(x: str) -> int:
    v = x.strip().lower()
    if v in {"1", "target", "true"}:
        return 1
    if v in {"0", "nontarget", "false", "non-target"}:
        return 0
    raise ValueError(f"Unsupported label: {x}")


def load_trials(path: Path) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                label = parse_label(parts[2])
                u1, u2 = parts[0], parts[1]
            except Exception:
                label = parse_label(parts[0])
                u1, u2 = parts[1], parts[2]
            rows.append((label, u1, u2))
    return rows


def fbank_one(wav: torch.Tensor) -> torch.Tensor:
    feat = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0.0)
    feat = feat - feat.mean(0, keepdim=True)
    return feat.unsqueeze(0)


def _add_audio_index_key(index: dict[str, str], key: str, wav_path: str) -> None:
    key = key.strip()
    if key and key not in index:
        index[key] = wav_path


def build_audio_index(eval_audio_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    supported_ext = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    for audio_path in eval_audio_root.rglob("*"):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in supported_ext:
            continue

        rel = audio_path.relative_to(eval_audio_root).as_posix()  # e.g. test/xxx.flac
        rel_no_ext = str(Path(rel).with_suffix(""))
        stem = audio_path.stem

        _add_audio_index_key(index, rel, str(audio_path))
        _add_audio_index_key(index, rel_no_ext, str(audio_path))
        _add_audio_index_key(index, stem, str(audio_path))

        if rel.startswith("enroll/"):
            _add_audio_index_key(index, rel[len("enroll/") :], str(audio_path))
            _add_audio_index_key(index, str(Path(rel[len("enroll/") :]).with_suffix("")), str(audio_path))
        if rel.startswith("test/"):
            _add_audio_index_key(index, rel[len("test/") :], str(audio_path))
            _add_audio_index_key(index, str(Path(rel[len("test/") :]).with_suffix("")), str(audio_path))

    return index


def resolve_wav_path(utt: str, audio_index: dict[str, str]) -> str | None:
    exts = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
    utt_path = Path(utt)
    utt_no_ext = str(utt_path.with_suffix(""))
    candidates = [utt, utt_no_ext, f"enroll/{utt}", f"test/{utt}", f"enroll/{utt_no_ext}", f"test/{utt_no_ext}"]
    for ext in exts:
        candidates.extend(
            [
                f"{utt}{ext}",
                f"enroll/{utt}{ext}",
                f"test/{utt}{ext}",
                f"{utt_no_ext}{ext}",
                f"enroll/{utt_no_ext}{ext}",
                f"test/{utt_no_ext}{ext}",
            ]
        )
    for key in candidates:
        if key in audio_index:
            return audio_index[key]
    return None


def extract_embeddings(
    utts: list[str],
    audio_index: dict[str, str],
    camp: FrozenCampPlus,
    device: torch.device,
    log_interval: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    embd: dict[str, np.ndarray] = {}
    missing: list[str] = []

    camp.eval()
    with torch.inference_mode():
        for idx, utt in enumerate(utts, start=1):
            wav_path = resolve_wav_path(utt, audio_index)
            if wav_path is None:
                missing.append(utt)
                continue

            wav = load_audio_mono(wav_path, sample_rate=16000)
            feat = fbank_one(wav).to(device)
            emb = camp(feat).squeeze(0)
            embd[utt] = emb.detach().cpu().numpy().astype(np.float32)

            if log_interval > 0 and idx % log_interval == 0:
                print(f"[embed] {idx}/{len(utts)}")

    return embd, missing


def score_trials(trials: list[tuple[int, str, str]], embd: dict[str, np.ndarray]) -> tuple[list[int], list[float], int]:
    labels: list[int] = []
    scores: list[float] = []
    missing_trials = 0

    for label, u1, u2 in trials:
        e1 = embd.get(u1)
        e2 = embd.get(u2)
        if e1 is None or e2 is None:
            missing_trials += 1
            continue
        score = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        labels.append(label)
        scores.append(score)

    return labels, scores, missing_trials


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pretrained CAMP++ on full CN-Celeb eval trials.")
    p.add_argument(
        "--campplus_ckpt",
        type=str,
        default="/home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin",
        help="Path to CAMP++ checkpoint bin.",
    )
    p.add_argument(
        "--cnceleb_root",
        type=str,
        default="/home/dgx/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac",
        help="CN-Celeb root directory.",
    )
    p.add_argument(
        "--trials_file",
        type=str,
        default="",
        help="Trials list path. Default: <cnceleb_root>/eval/lists/trials.lst",
    )
    p.add_argument(
        "--eval_audio_root",
        type=str,
        default="",
        help="Eval audio root. Default: <cnceleb_root>/eval",
    )
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--dry_run", action="store_true", help="Only check trials-to-audio mapping coverage and exit.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cnceleb_root = Path(args.cnceleb_root).resolve()
    trials_file = Path(args.trials_file).resolve() if args.trials_file else (cnceleb_root / "eval" / "lists" / "trials.lst")
    eval_audio_root = Path(args.eval_audio_root).resolve() if args.eval_audio_root else (cnceleb_root / "eval")

    if not trials_file.exists():
        raise FileNotFoundError(f"trials_file not found: {trials_file}")
    if not eval_audio_root.exists():
        raise FileNotFoundError(f"eval_audio_root not found: {eval_audio_root}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"[info] trials_file={trials_file}")
    print(f"[info] eval_audio_root={eval_audio_root}")
    print(f"[info] device={device}")

    trials = load_trials(trials_file)
    utt_set = sorted({u for _, u1, u2 in trials for u in (u1, u2)})
    print(f"[info] trials={len(trials)} unique_utts={len(utt_set)}")

    audio_index = build_audio_index(eval_audio_root)
    print(f"[info] indexed_wavs={len(audio_index)} (with aliases)")

    matched_utts = sum(1 for u in utt_set if resolve_wav_path(u, audio_index) is not None)
    print(f"[info] utt_mapping_coverage={matched_utts}/{len(utt_set)}")

    if args.dry_run:
        trial_miss = 0
        for _, u1, u2 in trials:
            if resolve_wav_path(u1, audio_index) is None or resolve_wav_path(u2, audio_index) is None:
                trial_miss += 1
        print(f"[info] trial_mapping_missing={trial_miss}/{len(trials)}")
        print("[info] dry_run done")
        return

    camp = FrozenCampPlus(args.campplus_ckpt).to(device)

    embd, missing_utts = extract_embeddings(
        utts=utt_set,
        audio_index=audio_index,
        camp=camp,
        device=device,
        log_interval=args.log_interval,
    )

    labels, scores, missing_trials = score_trials(trials, embd)
    if not labels:
        raise RuntimeError("No valid trials after scoring. Check trials/audio mapping.")

    eer, min_dcf = compute_eer_mindcf(labels, scores)

    result = {
        "dataset": "CN-Celeb eval",
        "campplus_ckpt": str(Path(args.campplus_ckpt).resolve()),
        "trials_file": str(trials_file),
        "eval_audio_root": str(eval_audio_root),
        "num_trials_total": int(len(trials)),
        "num_trials_scored": int(len(labels)),
        "num_trials_missing": int(missing_trials),
        "num_unique_utts": int(len(utt_set)),
        "num_utts_embedded": int(len(embd)),
        "num_utts_missing": int(len(missing_utts)),
        "eer_percent": float(eer),
        "min_dcf": float(min_dcf),
    }

    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[result] eer={eer:.4f} minDCF={min_dcf:.6f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
