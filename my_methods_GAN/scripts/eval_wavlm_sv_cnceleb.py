#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "my_methods_GAN"))

from speakerlab.utils.score_metrics import compute_eer, compute_pmiss_pfa_rbst
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


def _add_index(index: dict[str, str], key: str, path: str) -> None:
    key = key.strip()
    if key and key not in index:
        index[key] = path


def build_audio_index(eval_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    for p in eval_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(eval_root).as_posix()
        rel_no_ext = str(Path(rel).with_suffix(""))
        _add_index(index, rel, str(p))
        _add_index(index, rel_no_ext, str(p))
        _add_index(index, p.stem, str(p))
        if rel.startswith("enroll/"):
            tail = rel[len("enroll/") :]
            _add_index(index, tail, str(p))
            _add_index(index, str(Path(tail).with_suffix("")), str(p))
        if rel.startswith("test/"):
            tail = rel[len("test/") :]
            _add_index(index, tail, str(p))
            _add_index(index, str(Path(tail).with_suffix("")), str(p))
    return index


def resolve_audio(utt: str, index: dict[str, str]) -> str | None:
    exts = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
    no_ext = str(Path(utt).with_suffix(""))
    cands = [utt, no_ext, f"enroll/{utt}", f"test/{utt}", f"enroll/{no_ext}", f"test/{no_ext}"]
    for ext in exts:
        cands.extend([f"{utt}{ext}", f"{no_ext}{ext}", f"enroll/{utt}{ext}", f"test/{utt}{ext}", f"enroll/{no_ext}{ext}", f"test/{no_ext}{ext}"])
    for c in cands:
        if c in index:
            return index[c]
    return None


def compute_accuracy(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    _, eer_thr = compute_eer(fnr, fpr, scores)

    pred_eer = (scores >= float(eer_thr)).astype(np.int32)
    acc_eer = float((pred_eer == labels).mean())

    order = np.argsort(-scores)
    ss = scores[order]
    ll = labels[order]
    pos_total = int(ll.sum())
    neg_total = int(len(ll) - pos_total)
    tp = np.cumsum(ll == 1)
    fp = np.cumsum(ll == 0)
    acc = (tp + (neg_total - fp)) / len(ll)
    best_i = int(np.argmax(acc))
    return acc_eer, float(acc[best_i]), float(ss[best_i])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small CN-Celeb eval for microsoft/wavlm-base-plus-sv with EER/minDCF/accuracy.")
    p.add_argument("--model_dir", type=str, default="my_methods_GAN/pretrained/WavLM_SV")
    p.add_argument("--cnceleb_root", type=str, default="egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac")
    p.add_argument("--trials_file", type=str, default="")
    p.add_argument("--eval_audio_root", type=str, default="")
    p.add_argument("--max_trials", type=int, default=50000, help="Use at most N trials for a quick eval. <=0 means full trials.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    except Exception as exc:
        raise RuntimeError("Please install transformers first: pip install transformers") from exc

    root = Path(args.cnceleb_root).resolve()
    trials_file = Path(args.trials_file).resolve() if args.trials_file else (root / "eval" / "lists" / "trials.lst")
    eval_audio_root = Path(args.eval_audio_root).resolve() if args.eval_audio_root else (root / "eval")

    trials = load_trials(trials_file)
    if args.max_trials > 0 and len(trials) > args.max_trials:
        rng = random.Random(args.seed)
        trials = rng.sample(trials, args.max_trials)

    utts = sorted({u for _, a, b in trials for u in (a, b)})
    index = build_audio_index(eval_audio_root)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir).resolve()

    print(f"[info] trials={len(trials)} unique_utts={len(utts)} indexed={len(index)} device={device}")

    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    model = WavLMForXVector.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()

    embd: dict[str, np.ndarray] = {}
    missing_utts = 0
    with torch.inference_mode():
        for i, utt in enumerate(utts, start=1):
            wav_path = resolve_audio(utt, index)
            if wav_path is None:
                missing_utts += 1
                continue
            wav = load_audio_mono(wav_path, sample_rate=16000)
            inputs = feat_extractor([wav.numpy()], sampling_rate=16000, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model(**inputs).embeddings
            emb = torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
            embd[utt] = emb
            if args.log_interval > 0 and i % args.log_interval == 0:
                print(f"[embed] {i}/{len(utts)}")

    labels: list[int] = []
    scores: list[float] = []
    missing_trials = 0
    for y, u1, u2 in trials:
        e1 = embd.get(u1)
        e2 = embd.get(u2)
        if e1 is None or e2 is None:
            missing_trials += 1
            continue
        s = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        labels.append(y)
        scores.append(s)

    if not labels:
        raise RuntimeError("No valid scored trials. Check path mapping and model files.")

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)

    eer, min_dcf = compute_eer_mindcf(labels, scores)
    acc_eer, best_acc, best_thr = compute_accuracy(labels_np, scores_np)

    result = {
        "model_dir": str(model_dir),
        "trials_file": str(trials_file),
        "eval_audio_root": str(eval_audio_root),
        "num_trials_input": int(len(trials)),
        "num_trials_scored": int(len(labels)),
        "num_trials_missing": int(missing_trials),
        "num_unique_utts": int(len(utts)),
        "num_utts_embedded": int(len(embd)),
        "num_utts_missing": int(missing_utts),
        "eer_percent": float(eer),
        "min_dcf": float(min_dcf),
        "acc_at_eer_threshold": float(acc_eer),
        "best_accuracy": float(best_acc),
        "best_accuracy_threshold": float(best_thr),
    }

    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[result] eer={eer:.4f} minDCF={min_dcf:.6f} acc@eer_thr={acc_eer:.4f} best_acc={best_acc:.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
