#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sv_codec_restore_gan.models.campplus_wrapper import FrozenCampPlus
from sv_codec_restore_gan.models.ecapa_tdnn_wrapper import FrozenECAPATDNN
from sv_codec_restore_gan.models.generator import SVCodecRestoreGenerator
from sv_codec_restore_gan.utils.audio import load_audio_mono
from sv_codec_restore_gan.utils.metrics import compute_eer_mindcf


@dataclass(frozen=True)
class BackendConfig:
    name: str
    model: torch.nn.Module
    feature_type: str
    sample_rate: int = 16000
    feat_dim: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0


def build_backend(backend_type: str, backend_ckpt: str | Path) -> BackendConfig:
    backend_name = backend_type.strip().lower()
    ckpt_path = Path(backend_ckpt).resolve()
    if backend_name == "campplus":
        return BackendConfig(
            name="campplus",
            model=FrozenCampPlus(ckpt_path),
            feature_type="fbank",
        )
    if backend_name == "ecapa_tdnn":
        return BackendConfig(
            name="ecapa_tdnn",
            model=FrozenECAPATDNN(ckpt_path),
            # Keep feature extraction aligned with speakerlab's released
            # ECAPA inference pipeline for this checkpoint.
            feature_type="fbank",
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
        )
    raise ValueError(f"Unsupported backend_type: {backend_type}")


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


def _sample_without_replacement(
    rows: list[tuple[int, str, str]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, str, str]]:
    if count >= len(rows):
        return list(rows)
    if count <= 0:
        return []
    idx = rng.choice(len(rows), size=count, replace=False)
    return [rows[int(i)] for i in idx.tolist()]


def stratified_sample_trials(
    trials: list[tuple[int, str, str]],
    sample_fraction: float,
    sample_total: int,
    sample_pos_count: int,
    sample_neg_count: int,
    sample_seed: int,
) -> tuple[list[tuple[int, str, str]], dict[str, int | float | bool]]:
    total = len(trials)
    pos_rows = [x for x in trials if int(x[0]) == 1]
    neg_rows = [x for x in trials if int(x[0]) == 0]
    pos_total = len(pos_rows)
    neg_total = len(neg_rows)

    if total == 0:
        meta = {
            "enabled": False,
            "total": 0,
            "sampled": 0,
            "pos_total": 0,
            "neg_total": 0,
            "pos_sampled": 0,
            "neg_sampled": 0,
        }
        return [], meta

    frac = float(sample_fraction)
    sample_total = max(0, int(sample_total))
    sample_pos_count = max(0, int(sample_pos_count))
    sample_neg_count = max(0, int(sample_neg_count))

    use_sampling = (frac < 1.0) or (sample_total > 0) or (sample_pos_count > 0) or (sample_neg_count > 0)
    if not use_sampling:
        meta = {
            "enabled": False,
            "total": total,
            "sampled": total,
            "pos_total": pos_total,
            "neg_total": neg_total,
            "pos_sampled": pos_total,
            "neg_sampled": neg_total,
        }
        return trials, meta

    rng = np.random.default_rng(int(sample_seed))
    if sample_pos_count > 0 or sample_neg_count > 0:
        target_pos = min(sample_pos_count if sample_pos_count > 0 else pos_total, pos_total)
        target_neg = min(sample_neg_count if sample_neg_count > 0 else neg_total, neg_total)
    elif sample_total > 0:
        target_total = min(sample_total, total)
        pos_ratio = float(pos_total) / float(total)
        target_pos = int(round(target_total * pos_ratio))
        target_pos = min(max(0, target_pos), pos_total)
        target_neg = min(max(0, target_total - target_pos), neg_total)
        if target_pos + target_neg < target_total:
            room_pos = pos_total - target_pos
            add_pos = min(target_total - (target_pos + target_neg), room_pos)
            target_pos += add_pos
        if target_pos + target_neg < target_total:
            room_neg = neg_total - target_neg
            add_neg = min(target_total - (target_pos + target_neg), room_neg)
            target_neg += add_neg
    else:
        frac = min(max(frac, 0.0), 1.0)
        target_pos = min(pos_total, int(round(pos_total * frac)))
        target_neg = min(neg_total, int(round(neg_total * frac)))
        if frac > 0.0:
            if pos_total > 0 and target_pos == 0:
                target_pos = 1
            if neg_total > 0 and target_neg == 0:
                target_neg = 1

    sampled_pos = _sample_without_replacement(pos_rows, target_pos, rng)
    sampled_neg = _sample_without_replacement(neg_rows, target_neg, rng)
    sampled = sampled_pos + sampled_neg
    if sampled:
        rng.shuffle(sampled)

    meta = {
        "enabled": True,
        "total": total,
        "sampled": len(sampled),
        "pos_total": pos_total,
        "neg_total": neg_total,
        "pos_sampled": len(sampled_pos),
        "neg_sampled": len(sampled_neg),
    }
    return sampled, meta


def build_trial_sampling_cache_tag(args: argparse.Namespace, trial_meta: dict[str, int | float | bool]) -> str:
    if not bool(trial_meta.get("enabled", False)):
        return ""
    payload = {
        "trials_file": str(args.trials_file),
        "sample_fraction": float(args.trial_sample_fraction),
        "sample_total": int(args.trial_sample_total),
        "sample_pos_count": int(args.trial_sample_pos_count),
        "sample_neg_count": int(args.trial_sample_neg_count),
        "sample_seed": int(args.trial_sample_seed),
        "sampled": int(trial_meta.get("sampled", 0)),
        "total": int(trial_meta.get("total", 0)),
    }
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"__trialsub_{int(trial_meta.get('sampled', 0))}_{digest}"


def parse_conditions(text: str) -> list[str]:
    valid = {"clean", "coded", "restored"}
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not items:
        raise ValueError("--conditions is empty")

    dedup = []
    seen = set()
    for item in items:
        if item not in valid:
            raise ValueError(f"Unsupported condition: {item}")
        if item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup


def extract_feature_one(wav: torch.Tensor, backend: BackendConfig) -> torch.Tensor:
    kaldi_kwargs = {
        "sample_frequency": float(backend.sample_rate),
        "frame_length": float(backend.frame_length_ms),
        "frame_shift": float(backend.frame_shift_ms),
        "dither": 0.0,
    }
    if backend.feature_type == "fbank":
        feat = Kaldi.fbank(
            wav.unsqueeze(0),
            num_mel_bins=int(backend.feat_dim),
            **kaldi_kwargs,
        )
    elif backend.feature_type == "mfcc":
        feat = Kaldi.mfcc(
            wav.unsqueeze(0),
            num_ceps=int(backend.feat_dim),
            num_mel_bins=int(backend.feat_dim),
            **kaldi_kwargs,
        )
    else:
        raise ValueError(f"Unsupported feature_type: {backend.feature_type}")
    feat = feat - feat.mean(0, keepdim=True)
    return feat.unsqueeze(0)


class PlainFeatureDataset(Dataset):
    def __init__(self, items: list[tuple[str, str]], backend: BackendConfig):
        self.items = items
        self.backend = backend

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        utt, wav_path = self.items[idx]
        wav = load_audio_mono(wav_path, sample_rate=self.backend.sample_rate)
        feat = extract_feature_one(wav, self.backend).squeeze(0).contiguous()
        return utt, feat


def _collate_plain_feature(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], list[torch.Tensor]]:
    utts = [x[0] for x in batch]
    feats = [x[1] for x in batch]
    return utts, feats


def _run_backend_batch(
    backend: BackendConfig,
    device: torch.device,
    buckets: dict[int, list[tuple[str, torch.Tensor]]],
    out: dict[str, np.ndarray],
    pending: dict[str, np.ndarray],
    processed_utts: set[str],
    max_batch_size: int,
) -> None:
    for tlen in list(buckets.keys()):
        queue = buckets[tlen]
        while len(queue) >= max_batch_size:
            chunk = queue[:max_batch_size]
            del queue[:max_batch_size]
            utts = [x[0] for x in chunk]
            feats = torch.stack([x[1] for x in chunk], dim=0).to(device)
            emb = F.normalize(backend.model(feats), dim=-1)
            emb_np = emb.detach().cpu().numpy().astype(np.float32)
            for i, utt in enumerate(utts):
                out[utt] = emb_np[i]
                pending[utt] = emb_np[i]
                processed_utts.add(utt)


def _flush_backend_buckets(
    backend: BackendConfig,
    device: torch.device,
    buckets: dict[int, list[tuple[str, torch.Tensor]]],
    out: dict[str, np.ndarray],
    pending: dict[str, np.ndarray],
    processed_utts: set[str],
) -> None:
    for tlen in list(buckets.keys()):
        queue = buckets[tlen]
        if not queue:
            continue
        utts = [x[0] for x in queue]
        feats = torch.stack([x[1] for x in queue], dim=0).to(device)
        emb = F.normalize(backend.model(feats), dim=-1)
        emb_np = emb.detach().cpu().numpy().astype(np.float32)
        for i, utt in enumerate(utts):
            out[utt] = emb_np[i]
            pending[utt] = emb_np[i]
            processed_utts.add(utt)
        queue.clear()


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


def _init_embedding_state(
    mode: str,
    cache_path: Path | None,
    overwrite_cache: bool,
    cache_incremental: bool,
    cache_save_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], set[str], Path | None, int, int]:
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

    return out, pending, processed_utts, parts_dir, part_id, save_every


def _flush_incremental_cache(
    mode: str,
    pending: dict[str, np.ndarray],
    cache_path: Path | None,
    parts_dir: Path | None,
    part_id: int,
) -> int:
    if not pending or cache_path is None or parts_dir is None:
        return part_id
    part_id += 1
    part_path = _write_cache_part(parts_dir, part_id, pending)
    print(f"[{mode}] incremental cache part saved: {part_path} (items={len(pending)})")
    pending.clear()
    return part_id


def extract_emb_plain(
    utt2wav: dict[str, str],
    mode: str,
    backend: BackendConfig,
    device: torch.device,
    cache_path: Path | None,
    overwrite_cache: bool,
    cache_save_every: int,
    cache_incremental: bool,
    plain_loader_batch_size: int,
    plain_num_workers: int,
    plain_backend_batch_size: int,
) -> dict[str, np.ndarray]:
    if cache_path is not None and cache_path.exists() and not overwrite_cache:
        print(f"[{mode}] load embedding cache: {cache_path}")
        return load_embedding_cache(cache_path)

    out, pending, processed_utts, parts_dir, part_id, save_every = _init_embedding_state(
        mode=mode,
        cache_path=cache_path,
        overwrite_cache=overwrite_cache,
        cache_incremental=cache_incremental,
        cache_save_every=cache_save_every,
    )

    backend.model.eval()
    items = [(utt, wav_path) for utt, wav_path in utt2wav.items() if utt not in processed_utts]
    total = len(items)
    if total == 0:
        if cache_path is not None:
            save_embedding_cache(cache_path, out)
            print(f"[{mode}] save embedding cache: {cache_path}")
        return out

    loader_batch_size = max(1, int(plain_loader_batch_size))
    num_workers = max(0, int(plain_num_workers))
    backend_batch_size = max(1, int(plain_backend_batch_size))

    loader_kwargs = {
        "batch_size": loader_batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _collate_plain_feature,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(PlainFeatureDataset(items, backend), **loader_kwargs)

    buckets: dict[int, list[tuple[str, torch.Tensor]]] = {}
    next_report = 200
    with torch.inference_mode():
        for utts, feats in loader:
            for utt, feat in zip(utts, feats):
                tlen = int(feat.shape[0])
                buckets.setdefault(tlen, []).append((utt, feat))

            _run_backend_batch(
                backend=backend,
                device=device,
                buckets=buckets,
                out=out,
                pending=pending,
                processed_utts=processed_utts,
                max_batch_size=backend_batch_size,
            )

            processed_cnt = len(processed_utts)
            while processed_cnt >= next_report:
                print(f"[{mode}] {processed_cnt}/{total}")
                next_report += 200

            if cache_incremental and cache_path is not None and parts_dir is not None and len(pending) >= save_every:
                part_id = _flush_incremental_cache(mode, pending, cache_path, parts_dir, part_id)

        _flush_backend_buckets(
            backend=backend,
            device=device,
            buckets=buckets,
            out=out,
            pending=pending,
            processed_utts=processed_utts,
        )

    if cache_incremental and cache_path is not None and parts_dir is not None:
        part_id = _flush_incremental_cache(mode, pending, cache_path, parts_dir, part_id)

    if cache_path is not None:
        save_embedding_cache(cache_path, out)
        print(f"[{mode}] save embedding cache: {cache_path}")
    return out


def extract_emb_restored(
    utt2wav: dict[str, str],
    mode: str,
    generator: SVCodecRestoreGenerator,
    backend: BackendConfig,
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

    out, pending, processed_utts, parts_dir, part_id, save_every = _init_embedding_state(
        mode=mode,
        cache_path=cache_path,
        overwrite_cache=overwrite_cache,
        cache_incremental=cache_incremental,
        cache_save_every=cache_save_every,
    )

    generator.eval()
    backend.model.eval()
    with torch.inference_mode():
        for idx, (utt, wav_path) in enumerate(utt2wav.items(), start=1):
            if utt in processed_utts:
                continue

            wav = load_audio_mono(wav_path, sample_rate=backend.sample_rate)
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
                    sample_rate=backend.sample_rate,
                )
            else:
                wav = restore_in_chunks(
                    wav,
                    generator,
                    device,
                    chunk_seconds=restore_chunk_seconds,
                    overlap_seconds=restore_overlap_seconds,
                    chunk_batch_size=restore_chunk_batch_size,
                    sample_rate=backend.sample_rate,
                )

            feat = extract_feature_one(wav, backend).to(device)
            emb = backend.model(feat).squeeze(0)
            emb = F.normalize(emb, dim=-1)
            emb_np = emb.detach().cpu().numpy().astype(np.float32)
            out[utt] = emb_np
            pending[utt] = emb_np
            processed_utts.add(utt)

            del wav, feat, emb
            if idx % 200 == 0:
                print(f"[{mode}] {idx}/{len(utt2wav)}")

            if cache_incremental and cache_path is not None and parts_dir is not None and len(pending) >= save_every:
                part_id = _flush_incremental_cache(mode, pending, cache_path, parts_dir, part_id)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if cache_incremental and cache_path is not None and parts_dir is not None:
        part_id = _flush_incremental_cache(mode, pending, cache_path, parts_dir, part_id)

    if cache_path is not None:
        save_embedding_cache(cache_path, out)
        print(f"[{mode}] save embedding cache: {cache_path}")
    return out


def score_trials(trials, embd) -> tuple[list[int], list[float]]:
    valid_rows = [(label, u1, u2) for label, u1, u2 in trials if u1 in embd and u2 in embd]
    if not valid_rows:
        return [], []

    labels = np.asarray([row[0] for row in valid_rows], dtype=np.int32)
    e1 = np.stack([embd[row[1]] for row in valid_rows], axis=0).astype(np.float32, copy=False)
    e2 = np.stack([embd[row[2]] for row in valid_rows], axis=0).astype(np.float32, copy=False)
    denom = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    scores = np.sum(e1 * e2, axis=1) / np.clip(denom, 1e-8, None)
    return labels.tolist(), scores.astype(np.float64).tolist()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate clean/coded/restored SV metrics with configurable speaker backends.")
    p.add_argument("--clean_wav_scp", type=str, required=True)
    p.add_argument("--coded_wav_scp", type=str, required=True)
    p.add_argument("--trials_file", type=str, required=True)
    p.add_argument("--trial_sample_fraction", type=float, default=1.0, help="Stratified sample fraction on trials by label (0,1]. 1.0 means full trials.")
    p.add_argument("--trial_sample_total", type=int, default=0, help="Stratified sample total trial count. >0 overrides --trial_sample_fraction.")
    p.add_argument("--trial_sample_pos_count", type=int, default=0, help="Stratified sample positive trial count. >0 enables explicit per-class count sampling.")
    p.add_argument("--trial_sample_neg_count", type=int, default=0, help="Stratified sample negative trial count. >0 enables explicit per-class count sampling.")
    p.add_argument("--trial_sample_seed", type=int, default=42, help="Random seed for stratified trial sampling.")
    p.add_argument("--conditions", type=str, default="clean,coded,restored", help="Comma-separated eval conditions: clean,coded,restored")
    p.add_argument("--generator_ckpt", type=str, default="", help="Required only when conditions include restored.")
    p.add_argument("--backend_type", type=str, default="campplus", choices=["campplus", "ecapa_tdnn"])
    p.add_argument("--backend_ckpt", type=str, default="", help="Checkpoint path for the chosen speaker backend.")
    p.add_argument("--campplus_ckpt", type=str, default="", help="Backward-compatible alias for --backend_ckpt.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--restore_chunk_seconds", type=float, default=8.0, help="Chunk size (seconds) for restored inference. <=0 means full-utterance inference.")
    p.add_argument("--restore_overlap_seconds", type=float, default=0.1, help="Chunk overlap (seconds) for restored inference.")
    p.add_argument("--restore_chunk_batch_size", type=int, default=16, help="Batch size for restored chunk inference on GPU/CPU.")
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
    p.add_argument("--plain_loader_batch_size", type=int, default=64, help="DataLoader batch size for plain (clean/coded) feature extraction workers.")
    p.add_argument("--plain_num_workers", type=int, default=4, help="DataLoader worker count for plain (clean/coded) feature extraction.")
    p.add_argument("--plain_backend_batch_size", type=int, default=32, help="Backend batch size for plain branches (same-frame-length grouped).")
    p.add_argument("--plain_camp_batch_size", dest="plain_backend_batch_size", type=int, help="Backward-compatible alias for --plain_backend_batch_size.")
    p.add_argument("--dump_speaker_npy_dir", type=str, default="", help="If set, dump per-speaker embedding .npy files under <dir>/{clean,coded,restored}/")
    p.add_argument("--speaker_id_sep", type=str, default="/", help="Separator for parsing speaker id from utt key.")
    p.add_argument("--speaker_id_field", type=int, default=0, help="Field index after split for speaker id extraction.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    if not args.backend_ckpt:
        args.backend_ckpt = args.campplus_ckpt
    if not args.backend_ckpt:
        raise ValueError("Please provide --backend_ckpt, or use the legacy --campplus_ckpt alias.")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    conditions = parse_conditions(args.conditions)

    if args.trial_sample_fraction <= 0:
        raise ValueError("--trial_sample_fraction must be > 0")

    if args.overwrite_cache:
        print("[cache] --overwrite_cache is enabled: existing caches will be ignored and recomputed.")
    if not args.use_cache:
        print("[cache] --use_cache is disabled: all embeddings will be recomputed.")
    elif not args.cache_incremental:
        print("[cache] incremental cache is disabled; resume capability is off.")

    clean_scp = load_scp(args.clean_wav_scp)
    coded_scp = load_scp(args.coded_wav_scp)
    all_trials = load_trials(args.trials_file)
    trials, trial_meta = stratified_sample_trials(
        all_trials,
        sample_fraction=args.trial_sample_fraction,
        sample_total=args.trial_sample_total,
        sample_pos_count=args.trial_sample_pos_count,
        sample_neg_count=args.trial_sample_neg_count,
        sample_seed=args.trial_sample_seed,
    )
    if trial_meta["enabled"]:
        print(
            "[trials] stratified sampled: "
            f"total={trial_meta['total']} -> sampled={trial_meta['sampled']} "
            f"(pos {trial_meta['pos_sampled']}/{trial_meta['pos_total']}, "
            f"neg {trial_meta['neg_sampled']}/{trial_meta['neg_total']})"
        )

    if not trials:
        raise RuntimeError("No trials available after stratified sampling. Please adjust trial sampling arguments.")

    required_utts = {u for _, u1, u2 in trials for u in (u1, u2)}
    clean_scp_eval = {utt: wav for utt, wav in clean_scp.items() if utt in required_utts}
    coded_scp_eval = {utt: wav for utt, wav in coded_scp.items() if utt in required_utts}

    if "clean" in conditions:
        missing_clean = len(required_utts) - len(clean_scp_eval)
        if missing_clean > 0:
            print(f"[clean] warning: {missing_clean} utts in sampled trials are missing in clean scp")
    if "coded" in conditions or "restored" in conditions:
        missing_coded = len(required_utts) - len(coded_scp_eval)
        if missing_coded > 0:
            print(f"[coded/restored] warning: {missing_coded} utts in sampled trials are missing in coded scp")

    generator = None
    gen_tag = "none"
    if "restored" in conditions:
        if not args.generator_ckpt:
            raise ValueError("--generator_ckpt is required when --conditions includes restored")
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
        gen_tag = Path(args.generator_ckpt).stem

    backend = build_backend(args.backend_type, args.backend_ckpt)
    backend.model.to(device)

    cache_dir = None
    clean_cache = None
    coded_cache = None
    restored_cache = None
    if args.use_cache:
        cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else Path(args.output_json).resolve().parent / "emb_cache"
        clean_scp_tag = Path(args.clean_wav_scp).stem
        coded_scp_tag = Path(args.coded_wav_scp).stem
        backend_tag = Path(args.backend_ckpt).stem
        sample_tag = build_trial_sampling_cache_tag(args, trial_meta)
        clean_cache = cache_dir / f"clean__{clean_scp_tag}__backend_{args.backend_type}_{backend_tag}{sample_tag}.npz"
        coded_cache = cache_dir / f"coded__{coded_scp_tag}__backend_{args.backend_type}_{backend_tag}{sample_tag}.npz"
        restored_cache = cache_dir / f"restored__{coded_scp_tag}__gen_{gen_tag}__backend_{args.backend_type}_{backend_tag}{sample_tag}.npz"

    emb_by_mode: dict[str, dict[str, np.ndarray]] = {}
    if "clean" in conditions:
        emb_by_mode["clean"] = extract_emb_plain(
            clean_scp_eval,
            "clean",
            backend,
            device,
            cache_path=clean_cache,
            overwrite_cache=args.overwrite_cache,
            cache_save_every=args.cache_save_every,
            cache_incremental=args.cache_incremental,
            plain_loader_batch_size=args.plain_loader_batch_size,
            plain_num_workers=args.plain_num_workers,
            plain_backend_batch_size=args.plain_backend_batch_size,
        )
    if "coded" in conditions:
        emb_by_mode["coded"] = extract_emb_plain(
            coded_scp_eval,
            "coded",
            backend,
            device,
            cache_path=coded_cache,
            overwrite_cache=args.overwrite_cache,
            cache_save_every=args.cache_save_every,
            cache_incremental=args.cache_incremental,
            plain_loader_batch_size=args.plain_loader_batch_size,
            plain_num_workers=args.plain_num_workers,
            plain_backend_batch_size=args.plain_backend_batch_size,
        )
    if "restored" in conditions:
        emb_by_mode["restored"] = extract_emb_restored(
            coded_scp_eval,
            "restored",
            generator,
            backend,
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
    for name in conditions:
        emb = emb_by_mode[name]
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
        for mode in conditions:
            dump_speaker_embeddings(
                emb_by_mode[mode],
                mode=mode,
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
