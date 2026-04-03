#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sv_codec_restore_gan.models.campplus_wrapper import FrozenCampPlus
from sv_codec_restore_gan.models.generator import SVCodecRestoreGenerator
from sv_codec_restore_gan.utils.audio import load_audio_mono
from sv_codec_restore_gan.utils.metrics import compute_eer_mindcf


def load_scp(path: str | Path) -> dict[str, str]:
    out = {}
    with Path(path).resolve().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, wav = line.split(maxsplit=1)
            out[utt] = wav
    return out


def parse_label(x: str) -> int:
    v = x.strip().lower()
    if v in {"1", "target", "true"}:
        return 1
    if v in {"0", "nontarget", "false", "non-target"}:
        return 0
    raise ValueError(f"Unsupported label: {x}")


def load_trials(path: str | Path) -> list[tuple[int, str, str]]:
    rows = []
    with Path(path).resolve().open("r", encoding="utf-8") as f:
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


def extract_emb(
    utt2wav: dict[str, str],
    mode: str,
    generator: SVCodecRestoreGenerator,
    camp: FrozenCampPlus,
    device: torch.device,
) -> dict[str, np.ndarray]:
    out = {}
    generator.eval()
    camp.eval()
    with torch.no_grad():
        for idx, (utt, wav_path) in enumerate(utt2wav.items(), start=1):
            wav = load_audio_mono(wav_path, sample_rate=16000).to(device)
            if mode == "restored":
                wav = generator(wav.unsqueeze(0)).squeeze(0)
            feat = fbank_one(wav).to(device)
            emb = camp(feat).squeeze(0)
            emb = F.normalize(emb, dim=-1)
            out[utt] = emb.detach().cpu().numpy()
            if idx % 200 == 0:
                print(f"[{mode}] {idx}/{len(utt2wav)}")
    return out


def score_trials(trials, embd) -> tuple[list[int], list[float]]:
    labels, scores = [], []
    for label, u1, u2 in trials:
        if u1 not in embd or u2 not in embd:
            continue
        s = float(np.dot(embd[u1], embd[u2]) / (np.linalg.norm(embd[u1]) * np.linalg.norm(embd[u2]) + 1e-8))
        labels.append(label)
        scores.append(s)
    return labels, scores


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate clean/coded/restored SV metrics with CAM++ backend.")
    p.add_argument("--clean_wav_scp", type=str, required=True)
    p.add_argument("--coded_wav_scp", type=str, required=True)
    p.add_argument("--trials_file", type=str, required=True)
    p.add_argument("--generator_ckpt", type=str, required=True)
    p.add_argument("--campplus_ckpt", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    clean_scp = load_scp(args.clean_wav_scp)
    coded_scp = load_scp(args.coded_wav_scp)
    trials = load_trials(args.trials_file)

    ckpt = torch.load(str(Path(args.generator_ckpt).resolve()), map_location="cpu")
    model_args = ckpt.get("args", {})
    generator = SVCodecRestoreGenerator(
        emb_dim=int(model_args.get("emb_dim", 48)),
        num_blocks=int(model_args.get("num_blocks", 5)),
        hidden_units=int(model_args.get("hidden_units", 100)),
        attn_heads=int(model_args.get("attn_heads", 4)),
    )
    generator.load_state_dict(ckpt["generator"])
    generator.to(device)

    camp = FrozenCampPlus(args.campplus_ckpt).to(device)

    clean_emb = extract_emb(clean_scp, "clean", generator, camp, device)
    coded_emb = extract_emb(coded_scp, "coded", generator, camp, device)
    restored_emb = extract_emb(coded_scp, "restored", generator, camp, device)

    results = []
    for name, emb in [("clean", clean_emb), ("coded", coded_emb), ("restored", restored_emb)]:
        labels, scores = score_trials(trials, emb)
        eer, min_dcf = compute_eer_mindcf(labels, scores)
        results.append(
            {
                "condition": name,
                "num_trials": len(labels),
                "eer_percent": eer,
                "min_dcf": min_dcf,
            }
        )
        print(f"[{name}] trials={len(labels)} eer={eer:.4f} minDCF={min_dcf:.6f}")

    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
