#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def save_embedding_cache(path: Path, embd: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    utts = list(embd.keys())
    if not utts:
        np.savez_compressed(str(path), utt=np.array([], dtype=np.str_), emb=np.zeros((0, 0), dtype=np.float32))
        return

    emb = np.stack([np.asarray(embd[k], dtype=np.float32) for k in utts], axis=0)
    np.savez_compressed(str(path), utt=np.asarray(utts, dtype=np.str_), emb=emb)


def load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    utt = data["utt"]
    emb = data["emb"]
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return {str(k): emb[i] for i, k in enumerate(utt.tolist())}


def _cache_parts_dir(cache_path: Path) -> Path:
    return cache_path.parent / f"{cache_path.stem}_parts"


def _write_cache_part(parts_dir: Path, part_id: int, embd: dict[str, np.ndarray]) -> Path:
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = parts_dir / f"part_{part_id:06d}.npz"
    utts = list(embd.keys())
    emb = np.stack([np.asarray(embd[k], dtype=np.float32) for k in utts], axis=0)
    np.savez_compressed(str(part_path), utt=np.asarray(utts, dtype=np.str_), emb=emb)
    return part_path


def load_embedding_parts(parts_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not parts_dir.exists():
        return out
    for part in sorted(parts_dir.glob("part_*.npz")):
        data = np.load(str(part), allow_pickle=False)
        utt = data["utt"].tolist()
        emb = data["emb"]
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        for i, k in enumerate(utt):
            out[str(k)] = emb[i]
    return out


def _safe_name(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out).strip("_")
    return name or "unknown"


def _speaker_id_from_utt(utt: str, sep: str, field: int) -> str:
    if sep and sep in utt:
        parts = utt.split(sep)
        if -len(parts) <= field < len(parts):
            return parts[field]
    return utt


def dump_speaker_embeddings(
    embd: dict[str, np.ndarray],
    mode: str,
    dump_root: Path,
    speaker_id_sep: str,
    speaker_id_field: int,
) -> None:
    mode_dir = dump_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    bucket: dict[str, list[np.ndarray]] = {}
    for utt, emb in embd.items():
        spk = _speaker_id_from_utt(utt, speaker_id_sep, speaker_id_field)
        bucket.setdefault(spk, []).append(np.asarray(emb, dtype=np.float32))

    index = []
    for spk, vecs in bucket.items():
        mat = np.stack(vecs, axis=0)
        spk_emb = mat.mean(axis=0).astype(np.float32)
        spk_name = _safe_name(spk)
        spk_hash = hashlib.md5(spk.encode("utf-8")).hexdigest()[:8]
        filename = f"{spk_name}__{spk_hash}.npy"
        np.save(str(mode_dir / filename), spk_emb)
        index.append({"speaker_id": spk, "num_utts": int(mat.shape[0]), "file": filename})

    index_path = mode_dir / "speaker_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{mode}] dumped {len(index)} speaker npy files to: {mode_dir}")


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


def restore_in_chunks(
    wav: torch.Tensor,
    generator: SVCodecRestoreGenerator,
    device: torch.device,
    chunk_seconds: float,
    overlap_seconds: float,
    chunk_batch_size: int,
    sample_rate: int = 16000,
) -> torch.Tensor:
    chunk_len = int(chunk_seconds * sample_rate)
    overlap_len = int(overlap_seconds * sample_rate)
    if chunk_len <= 0 or wav.numel() <= chunk_len:
        return generator(wav.unsqueeze(0).to(device)).squeeze(0).detach().cpu()

    hop_len = max(1, chunk_len - overlap_len)
    chunk_batch_size = max(1, int(chunk_batch_size))
    restored = torch.zeros_like(wav)
    weight = torch.zeros_like(wav)

    pending_parts: list[torch.Tensor] = []
    pending_meta: list[tuple[int, int, int]] = []

    def _flush_pending() -> None:
        if not pending_parts:
            return
        batch = torch.stack(pending_parts, dim=0).to(device)
        out_batch = generator(batch).detach().cpu()
        for i, (start, end, valid_len) in enumerate(pending_meta):
            part_out = out_batch[i][:valid_len]
            restored[start:end] += part_out
            weight[start:end] += 1.0
        pending_parts.clear()
        pending_meta.clear()

    start = 0
    while start < wav.numel():
        end = min(start + chunk_len, wav.numel())
        part = wav[start:end]
        valid_len = part.numel()
        if valid_len < chunk_len:
            part = F.pad(part, (0, chunk_len - valid_len))

        pending_parts.append(part)
        pending_meta.append((start, end, valid_len))
        if len(pending_parts) >= chunk_batch_size:
            _flush_pending()
        start += hop_len

    _flush_pending()

    return restored / weight.clamp_min(1.0)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg and "memory" in msg


def restore_in_chunks_adaptive(
    wav: torch.Tensor,
    generator: SVCodecRestoreGenerator,
    device: torch.device,
    chunk_seconds: float,
    overlap_seconds: float,
    chunk_batch_size: int,
    min_chunk_seconds: float,
    chunk_shrink_factor: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    cur_chunk_seconds = float(chunk_seconds)
    min_chunk_seconds = max(0.2, float(min_chunk_seconds))
    chunk_shrink_factor = min(max(0.1, float(chunk_shrink_factor)), 0.95)

    while True:
        try:
            return restore_in_chunks(
                wav=wav,
                generator=generator,
                device=device,
                chunk_seconds=cur_chunk_seconds,
                overlap_seconds=overlap_seconds,
                chunk_batch_size=chunk_batch_size,
                sample_rate=sample_rate,
            )
        except RuntimeError as exc:
            if device.type != "cuda" or not _is_cuda_oom(exc):
                raise

            next_chunk_seconds = cur_chunk_seconds * chunk_shrink_factor
            if next_chunk_seconds < min_chunk_seconds:
                raise RuntimeError(
                    f"CUDA OOM while restored inference even after shrinking chunk to {cur_chunk_seconds:.2f}s; "
                    f"min_chunk_seconds={min_chunk_seconds:.2f}s"
                ) from exc

            print(
                f"[restored] CUDA OOM at chunk={cur_chunk_seconds:.2f}s, retry with {next_chunk_seconds:.2f}s"
            )
            torch.cuda.empty_cache()
            cur_chunk_seconds = next_chunk_seconds


def extract_emb(
    utt2wav: dict[str, str],
    mode: str,
    generator: SVCodecRestoreGenerator,
    camp: FrozenCampPlus,
    device: torch.device,
    restore_chunk_seconds: float,
    restore_overlap_seconds: float,
    restore_chunk_batch_size: int,
    restore_auto_shrink: bool,
    restore_min_chunk_seconds: float,
    restore_chunk_shrink_factor: float,
    cache_path: Path | None,
    overwrite_cache: bool,
    cache_save_every: int,
    cache_incremental: bool,
) -> dict[str, np.ndarray]:
    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        print(f"[{mode}] load embedding cache: {cache_path}")
        return load_embedding_cache(cache_path)

    out: dict[str, np.ndarray] = {}
    pending: dict[str, np.ndarray] = {}
    processed_utts: set[str] = set()
    part_id = 0
    save_every = max(1, int(cache_save_every))

    parts_dir = _cache_parts_dir(cache_path) if cache_path is not None else None
    if cache_incremental and cache_path is not None and parts_dir is not None and parts_dir.exists() and not overwrite_cache:
        out = load_embedding_parts(parts_dir)
        processed_utts = set(out.keys())
        existing_parts = sorted(parts_dir.glob("part_*.npz"))
        part_id = len(existing_parts)
        print(f"[{mode}] resume from incremental cache: {parts_dir} (loaded={len(out)})")

    generator.eval()
    camp.eval()
    with torch.inference_mode():
        for idx, (utt, wav_path) in enumerate(utt2wav.items(), start=1):
            if utt in processed_utts:
                continue

            wav = load_audio_mono(wav_path, sample_rate=16000)
            if mode == "restored":
                if restore_auto_shrink:
                    wav = restore_in_chunks_adaptive(
                        wav,
                        generator,
                        device,
                        chunk_seconds=restore_chunk_seconds,
                        overlap_seconds=restore_overlap_seconds,
                        chunk_batch_size=restore_chunk_batch_size,
                        min_chunk_seconds=restore_min_chunk_seconds,
                        chunk_shrink_factor=restore_chunk_shrink_factor,
                        sample_rate=16000,
                    )
                else:
                    wav = restore_in_chunks(
                        wav,
                        generator,
                        device,
                        chunk_seconds=restore_chunk_seconds,
                        overlap_seconds=restore_overlap_seconds,
                        chunk_batch_size=restore_chunk_batch_size,
                        sample_rate=16000,
                    )
            feat = fbank_one(wav).to(device)
            emb = camp(feat).squeeze(0)
            emb = F.normalize(emb, dim=-1)
            emb_np = emb.detach().cpu().numpy().astype(np.float32)
            out[utt] = emb_np
            pending[utt] = emb_np
            processed_utts.add(utt)

            del wav, feat, emb
            if idx % 200 == 0:
                print(f"[{mode}] {idx}/{len(utt2wav)}")

            if cache_incremental and cache_path is not None and parts_dir is not None and len(pending) >= save_every:
                part_id += 1
                part_path = _write_cache_part(parts_dir, part_id, pending)
                print(f"[{mode}] incremental cache part saved: {part_path} (items={len(pending)})")
                pending.clear()
                if device.type == "cuda" and mode == "restored":
                    torch.cuda.empty_cache()

    if cache_incremental and cache_path is not None and parts_dir is not None and pending:
        part_id += 1
        part_path = _write_cache_part(parts_dir, part_id, pending)
        print(f"[{mode}] incremental cache part saved: {part_path} (items={len(pending)})")
        pending.clear()

    if cache_path is not None:
        save_embedding_cache(cache_path, out)
        print(f"[{mode}] save embedding cache: {cache_path}")
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
    p.add_argument("--restore_chunk_seconds", type=float, default=8.0, help="Chunk size (seconds) for restored inference. <=0 means full-utterance inference.")
    p.add_argument("--restore_overlap_seconds", type=float, default=0.5, help="Chunk overlap (seconds) for restored inference.")
    p.add_argument("--restore_chunk_batch_size", type=int, default=8, help="Batch size for restored chunk inference on GPU/CPU.")
    p.add_argument("--restore_auto_shrink", action="store_true", help="Auto shrink chunk size and retry when restored inference hits CUDA OOM.")
    p.add_argument("--no_restore_auto_shrink", action="store_false", dest="restore_auto_shrink", help="Disable adaptive chunk shrinking on CUDA OOM.")
    p.set_defaults(restore_auto_shrink=True)
    p.add_argument("--restore_min_chunk_seconds", type=float, default=1.0, help="Minimum chunk seconds when auto shrinking is enabled.")
    p.add_argument("--restore_chunk_shrink_factor", type=float, default=0.7, help="OOM retry shrink factor for chunk seconds in (0,1).")
    p.add_argument("--use_cache", action="store_true", help="Enable local embedding cache for clean/coded/restored (default).")
    p.add_argument("--no_use_cache", action="store_false", dest="use_cache", help="Disable local embedding cache.")
    p.set_defaults(use_cache=True)
    p.add_argument("--cache_dir", type=str, default="", help="Cache directory. Default: <output_json_parent>/emb_cache")
    p.add_argument("--overwrite_cache", action="store_true", help="Recompute and overwrite existing embedding cache.")
    p.add_argument("--cache_save_every", type=int, default=500, help="Incremental cache save interval in utterances.")
    p.add_argument("--cache_incremental", action="store_true", help="Enable incremental cache write and resume.")
    p.add_argument("--no_cache_incremental", action="store_false", dest="cache_incremental", help="Disable incremental cache write and resume.")
    p.set_defaults(cache_incremental=True)
    p.add_argument("--dump_speaker_npy_dir", type=str, default="", help="If set, dump per-speaker embedding .npy files under <dir>/{clean,coded,restored}/")
    p.add_argument("--speaker_id_sep", type=str, default="/", help="Separator for parsing speaker id from utt key.")
    p.add_argument("--speaker_id_field", type=int, default=0, help="Field index after split for speaker id extraction.")
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

    cache_dir = None
    clean_cache = None
    coded_cache = None
    restored_cache = None
    if args.use_cache:
        cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else Path(args.output_json).resolve().parent / "emb_cache"
        clean_scp_tag = Path(args.clean_wav_scp).stem
        coded_scp_tag = Path(args.coded_wav_scp).stem
        camp_tag = Path(args.campplus_ckpt).stem
        gen_tag = Path(args.generator_ckpt).stem
        clean_cache = cache_dir / f"clean__{clean_scp_tag}__camp_{camp_tag}.npz"
        coded_cache = cache_dir / f"coded__{coded_scp_tag}__camp_{camp_tag}.npz"
        restored_cache = cache_dir / f"restored__{coded_scp_tag}__gen_{gen_tag}__camp_{camp_tag}.npz"

    clean_emb = extract_emb(
        clean_scp,
        "clean",
        generator,
        camp,
        device,
        restore_chunk_seconds=args.restore_chunk_seconds,
        restore_overlap_seconds=args.restore_overlap_seconds,
        restore_chunk_batch_size=args.restore_chunk_batch_size,
        restore_auto_shrink=args.restore_auto_shrink,
        restore_min_chunk_seconds=args.restore_min_chunk_seconds,
        restore_chunk_shrink_factor=args.restore_chunk_shrink_factor,
        cache_path=clean_cache,
        overwrite_cache=args.overwrite_cache,
        cache_save_every=args.cache_save_every,
        cache_incremental=args.cache_incremental,
    )
    coded_emb = extract_emb(
        coded_scp,
        "coded",
        generator,
        camp,
        device,
        restore_chunk_seconds=args.restore_chunk_seconds,
        restore_overlap_seconds=args.restore_overlap_seconds,
        restore_chunk_batch_size=args.restore_chunk_batch_size,
        restore_auto_shrink=args.restore_auto_shrink,
        restore_min_chunk_seconds=args.restore_min_chunk_seconds,
        restore_chunk_shrink_factor=args.restore_chunk_shrink_factor,
        cache_path=coded_cache,
        overwrite_cache=args.overwrite_cache,
        cache_save_every=args.cache_save_every,
        cache_incremental=args.cache_incremental,
    )
    restored_emb = extract_emb(
        coded_scp,
        "restored",
        generator,
        camp,
        device,
        restore_chunk_seconds=args.restore_chunk_seconds,
        restore_overlap_seconds=args.restore_overlap_seconds,
        restore_chunk_batch_size=args.restore_chunk_batch_size,
        restore_auto_shrink=args.restore_auto_shrink,
        restore_min_chunk_seconds=args.restore_min_chunk_seconds,
        restore_chunk_shrink_factor=args.restore_chunk_shrink_factor,
        cache_path=restored_cache,
        overwrite_cache=args.overwrite_cache,
        cache_save_every=args.cache_save_every,
        cache_incremental=args.cache_incremental,
    )

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

    if args.dump_speaker_npy_dir:
        dump_root = Path(args.dump_speaker_npy_dir).resolve()
        dump_speaker_embeddings(
            clean_emb,
            mode="clean",
            dump_root=dump_root,
            speaker_id_sep=args.speaker_id_sep,
            speaker_id_field=args.speaker_id_field,
        )
        dump_speaker_embeddings(
            coded_emb,
            mode="coded",
            dump_root=dump_root,
            speaker_id_sep=args.speaker_id_sep,
            speaker_id_field=args.speaker_id_field,
        )
        dump_speaker_embeddings(
            restored_emb,
            mode="restored",
            dump_root=dump_root,
            speaker_id_sep=args.speaker_id_sep,
            speaker_id_field=args.speaker_id_field,
        )

    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
