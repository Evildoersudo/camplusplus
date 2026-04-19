#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
    p.add_argument("--manifest_csv", type=str, required=True, help="CSV with clean_wav/codec_wav columns.")
    p.add_argument("--generator_ckpt", type=str, required=True, help="Experiment A checkpoint path.")
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


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    rules = parse_remap_rules(args.path_remap)
    metric_names = parse_metric_names(args.metrics)
    stoi_func, pesq_func = maybe_import_optional_metrics(metric_names)

    manifest_rows = load_manifest_rows(Path(args.manifest_csv).resolve())
    if args.codec_filter:
        filt = args.codec_filter.strip().lower()
        manifest_rows = [r for r in manifest_rows if str(r.get("codec_type", "")).lower() == filt]

    if args.max_utts > 0 and len(manifest_rows) > args.max_utts:
        rng = random.Random(int(args.seed))
        manifest_rows = rng.sample(manifest_rows, args.max_utts)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    generator = build_generator_from_ckpt(Path(args.generator_ckpt).resolve(), device)

    sums_coded: dict[str, float] = {}
    sums_restored: dict[str, float] = {}
    codec_buckets: dict[str, dict[str, dict[str, float] | int]] = {}
    processed = 0
    skipped = 0

    with torch.no_grad():
        for i, row in enumerate(manifest_rows, start=1):
            clean_p = remap_path(row["clean_wav"], repo_root, rules)
            coded_p = remap_path(row["codec_wav"], repo_root, rules)
            codec = str(row.get("codec_type", "unknown")).lower()

            if not clean_p.exists() or not coded_p.exists():
                skipped += 1
                continue

            clean = load_audio_mono(clean_p, sample_rate=16000)
            coded = load_audio_mono(coded_p, sample_rate=16000)
            n = min(clean.numel(), coded.numel())
            if n <= 160:
                skipped += 1
                continue

            clean = clean[:n].contiguous()
            coded = coded[:n].contiguous()
            restored = generator(coded.unsqueeze(0).to(device)).squeeze(0).detach().cpu()[:n]

            coded_metrics = compute_metrics(coded, clean, metric_names, stoi_func, pesq_func)
            restored_metrics = compute_metrics(restored, clean, metric_names, stoi_func, pesq_func)
            if not coded_metrics or not restored_metrics:
                skipped += 1
                continue

            _update_sums(sums_coded, coded_metrics)
            _update_sums(sums_restored, restored_metrics)

            bucket = codec_buckets.setdefault(codec, {"count": 0, "coded_sum": {}, "restored_sum": {}})
            bucket["count"] = int(bucket["count"]) + 1
            _update_sums(bucket["coded_sum"], coded_metrics)  # type: ignore[arg-type]
            _update_sums(bucket["restored_sum"], restored_metrics)  # type: ignore[arg-type]

            processed += 1
            if args.verbose_every > 0 and i % args.verbose_every == 0:
                print(f"[progress] {i}/{len(manifest_rows)} rows, processed={processed}, skipped={skipped}")

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
        "manifest_csv": str(Path(args.manifest_csv).resolve()),
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
