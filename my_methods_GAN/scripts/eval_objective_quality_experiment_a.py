#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import json
import multiprocessing as mp
import random
import re
import sys
import time
import wave
from pathlib import Path

import torch
import torch.nn.functional as F
from concurrent.futures.process import BrokenProcessPool

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sv_codec_restore_gan.models.generator import SVCodecRestoreGenerator
from sv_codec_restore_gan.models.losses import loss_complex_l1, loss_mrstft
from sv_codec_restore_gan.utils.audio import load_audio_mono


HIGHER_BETTER = {"si_sdr_db", "snr_db", "stoi", "pesq_wb"}
LOWER_BETTER = {"l1", "mse", "mrstft", "complex_l1"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate objective speech quality for Experiment A checkpoint "
        "(coded->clean baseline vs restored->clean)."
    )
    p.add_argument("--manifest_csv", type=str, default="", help="CSV with clean_wav/codec_wav columns.")
    p.add_argument("--clean_wav_scp", type=str, default="", help="Optional clean wav.scp for eval-set pairing.")
    p.add_argument("--coded_wav_scp", type=str, default="", help="Optional coded wav.scp for eval-set pairing.")
    p.add_argument(
        "--codec_name",
        type=str,
        default="",
        help="Optional codec_type name when using scp input (e.g., opus or amrwb).",
    )
    p.add_argument("--generator_ckpt", type=str, default="", help="Experiment A checkpoint path.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument(
        "--metrics",
        type=str,
        default="si_sdr_db,snr_db,l1,mse,mrstft,complex_l1",
        help="Comma-separated metrics.",
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max_utts", type=int, default=0, help="0 means use all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--codec_filter",
        type=str,
        default="",
        help="Filter rows by codec_type, e.g. opus or amrwb. Empty means no filter.",
    )
    p.add_argument(
        "--path_remap",
        action="append",
        default=[],
        help="Optional path remap OLD=NEW. Can be passed multiple times.",
    )
    p.add_argument("--verbose_every", type=int, default=200)
    p.add_argument(
        "--infer_chunk_seconds",
        type=float,
        default=8.0,
        help="Chunked generator inference window (seconds). <=0 disables chunking.",
    )
    p.add_argument(
        "--infer_hop_seconds",
        type=float,
        default=8.0,
        help="Chunked generator inference hop (seconds). Used when chunking is enabled.",
    )
    p.add_argument(
        "--infer_chunk_batch_size",
        type=int,
        default=8,
        help="Batch size for chunked generator inference.",
    )
    p.add_argument(
        "--metric_chunk_seconds",
        type=float,
        default=8.0,
        help="Chunked window for STFT-based metrics (mrstft/complex_l1). <=0 uses full utterance.",
    )
    p.add_argument(
        "--metric_hop_seconds",
        type=float,
        default=8.0,
        help="Hop for STFT-based metric chunking.",
    )
    p.add_argument(
        "--clear_cuda_every",
        type=int,
        default=200,
        help="Call torch.cuda.empty_cache() every N processed utterances (cuda only, <=0 disables).",
    )
    p.add_argument(
        "--gc_collect_every",
        type=int,
        default=0,
        help="Call gc.collect() every N processed utterances (<=0 disables, default disabled).",
    )
    p.add_argument(
        "--mp_metric_workers",
        type=int,
        default=0,
        help="Process workers for STOI/PESQ computation. <=0 disables multiprocessing.",
    )
    p.add_argument(
        "--mp_metric_prefetch",
        type=int,
        default=32,
        help="Max pending async jobs for STOI/PESQ multiprocessing.",
    )
    p.add_argument(
        "--mp_metric_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Start method for metric worker processes. Use spawn for better stability with CUDA parent process.",
    )
    p.add_argument(
        "--mp_metric_max_tasks_per_child",
        type=int,
        default=200,
        help="Recycle metric workers every N tasks to mitigate native-library instability/leaks. <=0 disables recycle.",
    )
    p.add_argument(
        "--disable_stoi",
        action="store_true",
        help="Disable STOI even if stoi is listed in --metrics.",
    )
    p.add_argument(
        "--disable_pesq_wb",
        action="store_true",
        help="Disable PESQ-WB even if pesq_wb is listed in --metrics.",
    )
    p.add_argument(
        "--restored_cache_dir",
        type=str,
        default="",
        help="Directory for cached restored wav files.",
    )
    p.add_argument(
        "--write_restored_cache",
        action="store_true",
        help="Write restored wav to --restored_cache_dir.",
    )
    p.add_argument(
        "--read_restored_cache",
        action="store_true",
        help="Read restored wav from --restored_cache_dir when cache exists.",
    )
    p.add_argument(
        "--read_restored_cache_only",
        action="store_true",
        help="Only read restored cache (skip item if cache missing), no generator fallback.",
    )
    p.add_argument(
        "--restore_only",
        action="store_true",
        help="Only run coded->restored and cache wavs, do not compute objective metrics.",
    )
    return p.parse_args()


def parse_metric_names(text: str) -> list[str]:
    names = [x.strip() for x in text.split(",") if x.strip()]
    if not names:
        raise ValueError("No metrics specified.")
    valid = HIGHER_BETTER | LOWER_BETTER | {"stoi", "pesq_wb"}
    for n in names:
        if n not in valid:
            raise ValueError(f"Unsupported metric: {n}")
    return names


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "clean_wav" not in row or "codec_wav" not in row:
                raise ValueError("manifest csv must include clean_wav and codec_wav columns.")
            rows.append(row)
    return rows


def load_scp(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, wav = line.split(maxsplit=1)
            out[key] = wav
    return out


def infer_codec_from_path(text: str) -> str:
    s = text.lower()
    if "amrwb" in s or "amr_wb" in s:
        return "amrwb"
    if "opus" in s:
        return "opus"
    if "aac" in s:
        return "aac"
    if "g711" in s and "mulaw" in s:
        return "g711mulaw"
    if "g711" in s and "alaw" in s:
        return "g711alaw"
    if "mulaw" in s:
        return "g711mulaw"
    if "alaw" in s:
        return "g711alaw"
    return "unknown"


def build_rows_from_scp(clean_scp: Path, coded_scp: Path, codec_name: str) -> list[dict[str, str]]:
    clean_map = load_scp(clean_scp)
    coded_map = load_scp(coded_scp)
    keys = sorted(set(clean_map.keys()) & set(coded_map.keys()))
    if not keys:
        raise RuntimeError("No shared utterance keys between clean_wav_scp and coded_wav_scp.")
    ctype = codec_name.strip().lower()
    rows = []
    for k in keys:
        codec_wav = coded_map[k]
        rows.append(
            {
                "utt_id": k,
                "clean_wav": clean_map[k],
                "codec_wav": codec_wav,
                "codec_type": ctype if ctype else infer_codec_from_path(codec_wav),
            }
        )
    return rows


def parse_remap_rules(raw_rules: list[str]) -> list[tuple[str, str]]:
    rules = []
    for item in raw_rules:
        if "=" not in item:
            raise ValueError(f"Invalid --path_remap rule (expect OLD=NEW): {item}")
        old, new = item.split("=", 1)
        old = old.strip().rstrip("/")
        new = new.strip().rstrip("/")
        if old and new:
            rules.append((old, new))
    return rules


def remap_path(path_text: str, repo_root: Path, rules: list[tuple[str, str]]) -> Path:
    p = Path(path_text)
    if p.exists():
        return p

    s = path_text
    for old, new in rules:
        if s.startswith(old):
            cand = Path(new + s[len(old) :])
            if cand.exists():
                return cand

    ws_prefix = "/workspace/camplusplus/"
    if s.startswith(ws_prefix):
        cand = repo_root / s[len(ws_prefix) :]
        if cand.exists():
            return cand

    return p


def _safe_cache_id(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    s = s.strip("_")
    return s or "utt"


def _build_cache_path(cache_root: Path, codec: str, row: dict[str, str], idx: int) -> Path:
    utt_id = str(row.get("utt_id", "")).strip()
    if not utt_id:
        clean_stem = Path(str(row.get("clean_wav", f"clean_{idx:08d}"))).stem
        coded_stem = Path(str(row.get("codec_wav", f"coded_{idx:08d}"))).stem
        utt_id = f"{clean_stem}_{coded_stem}_{idx:08d}"
    c = _safe_cache_id(codec if codec else "unknown")
    u = _safe_cache_id(utt_id)
    return cache_root / c / f"{u}.wav"


def _save_wav_pcm16(path: Path, wav: torch.Tensor, sample_rate: int = 16000) -> None:
    # Save mono float tensor as 16-bit PCM wav without external codec deps.
    x = wav.detach().cpu().flatten()
    x = x.clamp(-1.0, 1.0)
    pcm = (x * 32767.0).round().to(torch.int16).numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)


def si_sdr_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred - pred.mean()
    target = target - target.mean()
    target_energy = target.pow(2).sum().clamp_min(eps)
    scale = (pred * target).sum() / target_energy
    s_target = scale * target
    e_noise = pred - s_target
    ratio = s_target.pow(2).sum().clamp_min(eps) / e_noise.pow(2).sum().clamp_min(eps)
    return float(10.0 * torch.log10(ratio).item())


def snr_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    noise = target - pred
    return float(
        10.0 * torch.log10(target.pow(2).mean().clamp_min(eps) / noise.pow(2).mean().clamp_min(eps)).item()
    )


def build_generator_from_ckpt(ckpt_path: Path, device: torch.device) -> SVCodecRestoreGenerator:
    state = torch.load(str(ckpt_path), map_location="cpu")

    ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}
    model = SVCodecRestoreGenerator(
        emb_dim=int(ckpt_args.get("emb_dim", 64)),
        num_blocks=int(ckpt_args.get("num_blocks", 6)),
        hidden_units=int(ckpt_args.get("hidden_units", 128)),
        attn_heads=int(ckpt_args.get("attn_heads", 4)),
        cws_subbands=int(ckpt_args.get("cws_subbands", 3)),
        n_fft=int(ckpt_args.get("n_fft", 512)),
        hop_length=int(ckpt_args.get("hop_length", 128)),
        win_length=int(ckpt_args.get("win_length", 512)),
    )

    gen_state = state["generator"] if isinstance(state, dict) and "generator" in state else state
    model.load_state_dict(gen_state, strict=True)
    model.to(device).eval()
    return model


def maybe_import_optional_metrics(requested: list[str]) -> tuple[object | None, object | None]:
    stoi_func = None
    pesq_func = None
    if "stoi" in requested:
        try:
            from pystoi import stoi as stoi_func  # type: ignore
        except Exception:
            print("[warn] pystoi not available, metric stoi will be skipped.")
    if "pesq_wb" in requested:
        try:
            from pesq import pesq as pesq_func  # type: ignore
        except Exception:
            print("[warn] pesq not available, metric pesq_wb will be skipped.")
    return stoi_func, pesq_func


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: list[str],
    stoi_func,
    pesq_func,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if "si_sdr_db" in metrics:
        out["si_sdr_db"] = si_sdr_db(pred, target)
    if "snr_db" in metrics:
        out["snr_db"] = snr_db(pred, target)
    if "l1" in metrics:
        out["l1"] = float(F.l1_loss(pred, target).item())
    if "mse" in metrics:
        out["mse"] = float(F.mse_loss(pred, target).item())
    if "mrstft" in metrics:
        out["mrstft"] = float(loss_mrstft(pred.unsqueeze(0), target.unsqueeze(0)).item())
    if "complex_l1" in metrics:
        out["complex_l1"] = float(loss_complex_l1(pred.unsqueeze(0), target.unsqueeze(0)).item())

    if "stoi" in metrics and stoi_func is not None:
        out["stoi"] = float(stoi_func(target.detach().cpu().numpy(), pred.detach().cpu().numpy(), 16000, extended=False))
    if "pesq_wb" in metrics and pesq_func is not None:
        out["pesq_wb"] = float(pesq_func(16000, target.detach().cpu().numpy(), pred.detach().cpu().numpy(), "wb"))
    return out


def _update_sums(dst: dict[str, float], src: dict[str, float]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + float(v)


def _iter_slices(length: int, win: int, hop: int):
    if win <= 0 or hop <= 0 or length <= win:
        yield 0, length
        return
    s = 0
    while s < length:
        e = min(length, s + win)
        yield s, e
        if e >= length:
            break
        s += hop


def run_generator_chunked(
    generator: SVCodecRestoreGenerator,
    coded: torch.Tensor,
    device: torch.device,
    chunk_samples: int,
    hop_samples: int,
    chunk_batch_size: int,
) -> torch.Tensor:
    if chunk_samples <= 0 or hop_samples <= 0 or coded.numel() <= chunk_samples:
        return generator(coded.unsqueeze(0).to(device)).squeeze(0).detach().cpu()

    out = torch.zeros_like(coded)
    wsum = torch.zeros_like(coded)
    win = torch.hann_window(chunk_samples, device="cpu")
    pad = max(0, chunk_samples - hop_samples)
    if pad > 0:
        coded_pad = F.pad(coded, (pad, pad))
    else:
        coded_pad = coded

    total = coded_pad.numel()
    starts = list(range(0, max(1, total - chunk_samples + 1), hop_samples))
    if not starts:
        starts = [0]

    chunk_batch_size = max(1, int(chunk_batch_size))
    for bi in range(0, len(starts), chunk_batch_size):
        batch_starts = starts[bi : bi + chunk_batch_size]
        chunks = [coded_pad[s : s + chunk_samples] for s in batch_starts]
        batch = torch.stack(chunks, dim=0).to(device)
        preds = generator(batch).detach().cpu()
        for j, s in enumerate(batch_starts):
            pred = preds[j] * win
            # map to original (unpadded) coordinate
            os = s - pad
            oe = os + chunk_samples
            cs = 0
            ce = chunk_samples
            if os < 0:
                cs = -os
                os = 0
            if oe > coded.numel():
                ce -= oe - coded.numel()
                oe = coded.numel()
            if os < oe and cs < ce:
                out[os:oe] += pred[cs:ce]
                wsum[os:oe] += win[cs:ce]

    # Fallback for any uncovered samples.
    mask = wsum <= 1e-8
    out = out / wsum.clamp_min(1e-8)
    if mask.any():
        out[mask] = coded[mask]
    return out


def _maybe_compute_optional_metrics(
    metrics: list[str],
    sr: int,
    target_np,
    pred_np,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if "stoi" in metrics:
        try:
            from pystoi import stoi as _stoi  # type: ignore

            out["stoi"] = float(_stoi(target_np, pred_np, sr, extended=False))
        except Exception:
            pass
    if "pesq_wb" in metrics:
        try:
            from pesq import pesq as _pesq  # type: ignore

            out["pesq_wb"] = float(_pesq(sr, target_np, pred_np, "wb"))
        except Exception:
            pass
    return out


def _compute_optional_metric_pair(
    metrics: list[str],
    sr: int,
    target_np,
    coded_np,
    restored_np,
) -> tuple[dict[str, float], dict[str, float]]:
    coded_opt = _maybe_compute_optional_metrics(metrics, sr, target_np, coded_np)
    restored_opt = _maybe_compute_optional_metrics(metrics, sr, target_np, restored_np)
    return coded_opt, restored_opt


def compute_stft_metrics_chunked(
    pred: torch.Tensor,
    target: torch.Tensor,
    want_mrstft: bool,
    want_complex: bool,
    chunk_samples: int,
    hop_samples: int,
) -> dict[str, float]:
    vals_mrstft: list[float] = []
    vals_complex: list[float] = []
    # torch.stft uses reflect-padding by default, so chunks that are too short can crash.
    # For mrstft (max n_fft=1024), need len > 512; for complex_l1 (n_fft=512), need len > 256.
    min_len_mrstft = 513
    min_len_complex = 257
    for s, e in _iter_slices(pred.numel(), chunk_samples, hop_samples):
        p = pred[s:e]
        t = target[s:e]
        if want_mrstft and p.numel() >= min_len_mrstft:
            vals_mrstft.append(float(loss_mrstft(p.unsqueeze(0), t.unsqueeze(0)).item()))
        if want_complex and p.numel() >= min_len_complex:
            vals_complex.append(float(loss_complex_l1(p.unsqueeze(0), t.unsqueeze(0)).item()))

    out: dict[str, float] = {}
    # Fallback for very short utterances/chunks: right-pad once and compute to keep metric present.
    if want_mrstft and not vals_mrstft:
        p = pred
        t = target
        if p.numel() < min_len_mrstft:
            pad = min_len_mrstft - p.numel()
            p = F.pad(p, (0, pad))
            t = F.pad(t, (0, pad))
        vals_mrstft.append(float(loss_mrstft(p.unsqueeze(0), t.unsqueeze(0)).item()))
    if want_complex and not vals_complex:
        p = pred
        t = target
        if p.numel() < min_len_complex:
            pad = min_len_complex - p.numel()
            p = F.pad(p, (0, pad))
            t = F.pad(t, (0, pad))
        vals_complex.append(float(loss_complex_l1(p.unsqueeze(0), t.unsqueeze(0)).item()))

    if want_mrstft and vals_mrstft:
        out["mrstft"] = float(sum(vals_mrstft) / len(vals_mrstft))
    if want_complex and vals_complex:
        out["complex_l1"] = float(sum(vals_complex) / len(vals_complex))
    return out


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    rules = parse_remap_rules(args.path_remap)
    metric_names = parse_metric_names(args.metrics)
    if args.disable_stoi:
        metric_names = [m for m in metric_names if m != "stoi"]
    if args.disable_pesq_wb:
        metric_names = [m for m in metric_names if m != "pesq_wb"]
    if not metric_names:
        raise ValueError("No metrics left after applying disable flags.")

    stoi_func, pesq_func = maybe_import_optional_metrics(metric_names)

    use_manifest = bool(args.manifest_csv.strip())
    use_scp = bool(args.clean_wav_scp.strip() or args.coded_wav_scp.strip())
    if use_manifest and use_scp:
        raise ValueError("Use either --manifest_csv OR (--clean_wav_scp and --coded_wav_scp), not both.")
    if not use_manifest and not use_scp:
        raise ValueError("Missing input. Provide --manifest_csv or both --clean_wav_scp/--coded_wav_scp.")
    if use_scp and (not args.clean_wav_scp.strip() or not args.coded_wav_scp.strip()):
        raise ValueError("When using scp input, both --clean_wav_scp and --coded_wav_scp are required.")

    if args.read_restored_cache_only and not args.read_restored_cache:
        raise ValueError("--read_restored_cache_only requires --read_restored_cache.")

    cache_root = Path(args.restored_cache_dir).resolve() if args.restored_cache_dir.strip() else None
    if (args.write_restored_cache or args.read_restored_cache or args.restore_only) and cache_root is None:
        raise ValueError("--restored_cache_dir is required when using cache/read/restore-only flags.")
    if args.restore_only and not args.write_restored_cache:
        raise ValueError("--restore_only requires --write_restored_cache.")
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    if use_manifest:
        manifest_rows = load_manifest_rows(Path(args.manifest_csv).resolve())
        input_source = {"type": "manifest_csv", "manifest_csv": str(Path(args.manifest_csv).resolve())}
    else:
        manifest_rows = build_rows_from_scp(
            Path(args.clean_wav_scp).resolve(),
            Path(args.coded_wav_scp).resolve(),
            args.codec_name,
        )
        input_source = {
            "type": "scp_pair",
            "clean_wav_scp": str(Path(args.clean_wav_scp).resolve()),
            "coded_wav_scp": str(Path(args.coded_wav_scp).resolve()),
        }
    if args.codec_filter:
        filt = args.codec_filter.strip().lower()
        manifest_rows = [r for r in manifest_rows if str(r.get("codec_type", "")).lower() == filt]

    if args.max_utts > 0 and len(manifest_rows) > args.max_utts:
        rng = random.Random(int(args.seed))
        manifest_rows = rng.sample(manifest_rows, args.max_utts)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    need_generator = bool(args.restore_only or (not args.read_restored_cache_only))
    generator = None
    if need_generator:
        if not args.generator_ckpt.strip():
            raise ValueError("--generator_ckpt is required unless running strictly with --read_restored_cache_only.")
        generator = build_generator_from_ckpt(Path(args.generator_ckpt).resolve(), device)

    sums_coded: dict[str, float] = {}
    sums_restored: dict[str, float] = {}
    codec_buckets: dict[str, dict[str, dict[str, float] | int]] = {}
    processed = 0
    skipped = 0
    t0 = time.time()

    print(
        f"[start] rows={len(manifest_rows)} device={device} "
        f"infer_chunk={args.infer_chunk_seconds}s metric_chunk={args.metric_chunk_seconds}s"
    )

    mp_metric_workers = max(0, int(args.mp_metric_workers))
    mp_metric_prefetch = max(1, int(args.mp_metric_prefetch))
    optional_metrics = [m for m in metric_names if m in {"stoi", "pesq_wb"}]
    metric_executor = None
    pending_metric_jobs: list[tuple[concurrent.futures.Future, str, dict[str, float], dict[str, float], int]] = []
    optional_metric_pool_broken = False
    if optional_metrics and mp_metric_workers > 0:
        mp_ctx = mp.get_context(args.mp_metric_start_method)
        max_tasks_per_child = int(args.mp_metric_max_tasks_per_child)
        if max_tasks_per_child > 0:
            metric_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=mp_metric_workers,
                mp_context=mp_ctx,
                max_tasks_per_child=max_tasks_per_child,
            )
        else:
            metric_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=mp_metric_workers,
                mp_context=mp_ctx,
            )
        print(
            f"[speedup] optional metrics multiprocessing enabled: workers={mp_metric_workers}, "
            f"prefetch={mp_metric_prefetch}, start_method={args.mp_metric_start_method}, "
            f"max_tasks_per_child={max_tasks_per_child if max_tasks_per_child > 0 else 'disabled'}"
        )

    def _finalize_one(codec: str, coded_metrics: dict[str, float], restored_metrics: dict[str, float], idx: int) -> None:
        nonlocal processed
        _update_sums(sums_coded, coded_metrics)
        _update_sums(sums_restored, restored_metrics)

        bucket = codec_buckets.setdefault(codec, {"count": 0, "coded_sum": {}, "restored_sum": {}})
        bucket["count"] = int(bucket["count"]) + 1
        _update_sums(bucket["coded_sum"], coded_metrics)  # type: ignore[arg-type]
        _update_sums(bucket["restored_sum"], restored_metrics)  # type: ignore[arg-type]

        processed += 1
        if args.verbose_every > 0 and idx % args.verbose_every == 0:
            elapsed = max(1e-6, time.time() - t0)
            speed = idx / elapsed
            remain = max(0, len(manifest_rows) - idx)
            eta = remain / max(1e-6, speed)
            print(
                "[progress] "
                f"{idx}/{len(manifest_rows)} ({100.0 * idx / len(manifest_rows):.1f}%) "
                f"processed={processed} skipped={skipped} "
                f"speed={speed:.2f} utt/s eta={eta:.1f}s"
            )

    def _drain_one_pending() -> None:
        nonlocal metric_executor, optional_metric_pool_broken
        if not pending_metric_jobs:
            return
        fut, codec, coded_metrics, restored_metrics, idx = pending_metric_jobs.pop(0)
        try:
            coded_opt, restored_opt = fut.result()
            coded_metrics.update(coded_opt)
            restored_metrics.update(restored_opt)
        except BrokenProcessPool as e:
            optional_metric_pool_broken = True
            print(
                "[warn] optional metric process pool crashed (BrokenProcessPool). "
                "Disable optional metrics (stoi/pesq_wb) for remaining utterances. "
                f"detail={e}"
            )
            if metric_executor is not None:
                try:
                    metric_executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                metric_executor = None
            # Drop queued optional jobs and continue with base metrics only.
            pending_metric_jobs.clear()
        except Exception as e:
            optional_metric_pool_broken = True
            print(
                "[warn] optional metric async job failed. "
                "Disable optional metrics (stoi/pesq_wb) for remaining utterances. "
                f"detail={e}"
            )
            if metric_executor is not None:
                try:
                    metric_executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                metric_executor = None
            pending_metric_jobs.clear()
        _finalize_one(codec, coded_metrics, restored_metrics, idx)

    with torch.no_grad():
        for i, row in enumerate(manifest_rows, start=1):
            clean_p = remap_path(row["clean_wav"], repo_root, rules)
            coded_p = remap_path(row["codec_wav"], repo_root, rules)
            codec = str(row.get("codec_type", "unknown")).lower()

            coded = load_audio_mono(coded_p, sample_rate=16000)
            if args.restore_only:
                if coded.numel() <= 160:
                    skipped += 1
                    continue
                chunk_samples = int(max(0.0, float(args.infer_chunk_seconds)) * 16000.0)
                hop_samples = int(max(0.0, float(args.infer_hop_seconds)) * 16000.0)
                assert generator is not None
                restored = run_generator_chunked(
                    generator,
                    coded.contiguous(),
                    device,
                    chunk_samples,
                    hop_samples,
                    chunk_batch_size=max(1, int(args.infer_chunk_batch_size)),
                )
                if cache_root is not None and args.write_restored_cache:
                    cache_path = _build_cache_path(cache_root, codec, row, i)
                    _save_wav_pcm16(cache_path, restored, sample_rate=16000)

                processed += 1
                if args.verbose_every > 0 and i % args.verbose_every == 0:
                    elapsed = max(1e-6, time.time() - t0)
                    speed = i / elapsed
                    remain = max(0, len(manifest_rows) - i)
                    eta = remain / max(1e-6, speed)
                    print(
                        "[progress] "
                        f"{i}/{len(manifest_rows)} ({100.0 * i / len(manifest_rows):.1f}%) "
                        f"cached={processed} skipped={skipped} speed={speed:.2f} utt/s eta={eta:.1f}s"
                    )
                if device.type == "cuda" and args.clear_cuda_every > 0 and processed % int(args.clear_cuda_every) == 0:
                    torch.cuda.empty_cache()
                if args.gc_collect_every > 0 and processed % int(args.gc_collect_every) == 0:
                    gc.collect()
                del coded, restored
                continue

            if not clean_p.exists() or not coded_p.exists():
                skipped += 1
                continue

            clean = load_audio_mono(clean_p, sample_rate=16000)
            n = min(clean.numel(), coded.numel())
            if n <= 160:
                skipped += 1
                continue

            clean = clean[:n].contiguous()
            coded = coded[:n].contiguous()
            cache_path = _build_cache_path(cache_root, codec, row, i) if cache_root is not None else None
            restored = None
            if args.read_restored_cache and cache_path is not None and cache_path.exists():
                restored = load_audio_mono(cache_path, sample_rate=16000)
                if restored.numel() < n:
                    restored = F.pad(restored, (0, n - restored.numel()))
                restored = restored[:n].contiguous()
            elif args.read_restored_cache_only:
                skipped += 1
                del clean, coded
                continue
            else:
                chunk_samples = int(max(0.0, float(args.infer_chunk_seconds)) * 16000.0)
                hop_samples = int(max(0.0, float(args.infer_hop_seconds)) * 16000.0)
                assert generator is not None
                restored = run_generator_chunked(
                    generator,
                    coded,
                    device,
                    chunk_samples,
                    hop_samples,
                    chunk_batch_size=max(1, int(args.infer_chunk_batch_size)),
                )[:n]
                if args.write_restored_cache and cache_path is not None:
                    _save_wav_pcm16(cache_path, restored, sample_rate=16000)

            assert restored is not None

            light_metrics = [m for m in metric_names if m not in {"mrstft", "complex_l1", "stoi", "pesq_wb"}]
            coded_metrics = compute_metrics(coded, clean, light_metrics, stoi_func, pesq_func)
            restored_metrics = compute_metrics(restored, clean, light_metrics, stoi_func, pesq_func)
            # Replace heavy STFT metrics with chunked version to reduce peak memory.
            if "mrstft" in metric_names or "complex_l1" in metric_names:
                mchunk = int(max(0.0, float(args.metric_chunk_seconds)) * 16000.0)
                mhop = int(max(0.0, float(args.metric_hop_seconds)) * 16000.0)
                coded_stft = compute_stft_metrics_chunked(
                    coded,
                    clean,
                    want_mrstft=("mrstft" in metric_names),
                    want_complex=("complex_l1" in metric_names),
                    chunk_samples=mchunk,
                    hop_samples=mhop,
                )
                restored_stft = compute_stft_metrics_chunked(
                    restored,
                    clean,
                    want_mrstft=("mrstft" in metric_names),
                    want_complex=("complex_l1" in metric_names),
                    chunk_samples=mchunk,
                    hop_samples=mhop,
                )
                coded_metrics.update(coded_stft)
                restored_metrics.update(restored_stft)

            if optional_metrics and not optional_metric_pool_broken:
                clean_np = clean.detach().cpu().numpy()
                coded_np = coded.detach().cpu().numpy()
                restored_np = restored.detach().cpu().numpy()
                if metric_executor is not None:
                    fut = metric_executor.submit(
                        _compute_optional_metric_pair,
                        optional_metrics,
                        16000,
                        clean_np,
                        coded_np,
                        restored_np,
                    )
                    pending_metric_jobs.append((fut, codec, coded_metrics, restored_metrics, i))
                else:
                    try:
                        coded_opt, restored_opt = _compute_optional_metric_pair(
                            optional_metrics, 16000, clean_np, coded_np, restored_np
                        )
                        coded_metrics.update(coded_opt)
                        restored_metrics.update(restored_opt)
                    except Exception as e:
                        optional_metric_pool_broken = True
                        print(
                            "[warn] optional metric sync compute failed. "
                            "Disable optional metrics (stoi/pesq_wb) for remaining utterances. "
                            f"detail={e}"
                        )

            if not coded_metrics or not restored_metrics:
                skipped += 1
                continue

            if metric_executor is not None and optional_metrics:
                while len(pending_metric_jobs) >= mp_metric_prefetch:
                    _drain_one_pending()
            else:
                _finalize_one(codec, coded_metrics, restored_metrics, i)

            if device.type == "cuda" and args.clear_cuda_every > 0 and processed % int(args.clear_cuda_every) == 0:
                torch.cuda.empty_cache()
            if args.gc_collect_every > 0 and processed % int(args.gc_collect_every) == 0:
                gc.collect()
            del clean, coded, restored, coded_metrics, restored_metrics

    if metric_executor is not None:
        while pending_metric_jobs:
            _drain_one_pending()
        metric_executor.shutdown(wait=True)

    if args.restore_only:
        result = {
            "mode": "restore_only",
            "input_source": input_source,
            "generator_ckpt": str(Path(args.generator_ckpt).resolve()) if args.generator_ckpt.strip() else "",
            "device_used": str(device),
            "num_rows_after_filter": len(manifest_rows),
            "cached": processed,
            "skipped": skipped,
            "restored_cache_dir": str(cache_root) if cache_root is not None else "",
        }
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("[done] restore-only caching finished.")
        print(f"[done] cached={processed}, skipped={skipped}")
        print(f"[done] output_json={out_path}")
        return

    if processed == 0:
        raise RuntimeError("No valid utterances were processed. Check paths/manifest/filter settings.")

    def _avg(d: dict[str, float], count: int) -> dict[str, float]:
        return {k: float(v) / float(count) for k, v in d.items()}

    mean_coded = _avg(sums_coded, processed)
    mean_restored = _avg(sums_restored, processed)

    improvement: dict[str, float] = {}
    for k in mean_coded.keys():
        if k in HIGHER_BETTER:
            improvement[k] = mean_restored[k] - mean_coded[k]
        elif k in LOWER_BETTER:
            improvement[k] = mean_coded[k] - mean_restored[k]
        else:
            improvement[k] = mean_restored[k] - mean_coded[k]

    by_codec = {}
    for codec, bucket in sorted(codec_buckets.items()):
        count = int(bucket["count"])
        coded_mean = _avg(bucket["coded_sum"], count)  # type: ignore[arg-type]
        restored_mean = _avg(bucket["restored_sum"], count)  # type: ignore[arg-type]
        codec_impr = {}
        for k in coded_mean.keys():
            if k in HIGHER_BETTER:
                codec_impr[k] = restored_mean[k] - coded_mean[k]
            elif k in LOWER_BETTER:
                codec_impr[k] = coded_mean[k] - restored_mean[k]
            else:
                codec_impr[k] = restored_mean[k] - coded_mean[k]
        by_codec[codec] = {
            "count": count,
            "coded_mean": coded_mean,
            "restored_mean": restored_mean,
            "improvement": codec_impr,
        }

    result = {
        "input_source": input_source,
        "generator_ckpt": str(Path(args.generator_ckpt).resolve()),
        "device_used": str(device),
        "metrics": metric_names,
        "num_rows_after_filter": len(manifest_rows),
        "processed": processed,
        "skipped": skipped,
        "overall": {
            "coded_mean": mean_coded,
            "restored_mean": mean_restored,
            "improvement": improvement,
            "improvement_note": "higher-better metrics use restored-coded; lower-better metrics use coded-restored",
        },
        "by_codec": by_codec,
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("[done] objective quality evaluation finished.")
    print(f"[done] processed={processed}, skipped={skipped}")
    print(f"[done] output_json={out_path}")
    print("[overall] coded_mean:", mean_coded)
    print("[overall] restored_mean:", mean_restored)
    print("[overall] improvement:", improvement)


if __name__ == "__main__":
    main()
