#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sv_codec_restore_gan.utils.metrics import compute_eer_mindcf

from eval_sv_codec_restore_gan import (
    build_backend,
    build_trial_sampling_cache_tag,
    dump_speaker_embeddings,
    extract_emb_plain,
    load_scp,
    load_trials,
    score_trials,
    stratified_sample_trials,
)


def parse_codec_scp_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen = set()
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid codec_scp entry: {item}. Expected format: codec_tag=path/to/scp")
        tag, scp = item.split("=", 1)
        tag = tag.strip()
        scp = scp.strip()
        if not tag or not scp:
            raise ValueError(f"Invalid codec_scp entry: {item}. Empty tag or path.")
        if tag in seen:
            raise ValueError(f"Duplicated codec tag: {tag}")
        seen.add(tag)
        entries.append((tag, scp))
    if not entries:
        raise ValueError("--codec_scp is empty")
    return entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluate coded-only SV metrics for multiple codec scp files.")
    p.add_argument("--codec_scp", type=str, required=True, help="Comma-separated codec_tag=scp_path entries.")
    p.add_argument("--trials_file", type=str, required=True)
    p.add_argument("--backend_type", type=str, default="campplus", choices=["campplus", "ecapa_tdnn", "eres2net"])
    p.add_argument("--backend_ckpt", type=str, default="", help="Checkpoint path for the chosen speaker backend.")
    p.add_argument("--campplus_ckpt", type=str, default="", help="Backward-compatible alias for --backend_ckpt.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument(
        "--output_png",
        type=str,
        default="",
        help="Optional summary figure path. Default: <output_json_stem>.png",
    )

    p.add_argument("--trial_sample_fraction", type=float, default=1.0)
    p.add_argument("--trial_sample_total", type=int, default=0)
    p.add_argument("--trial_sample_pos_count", type=int, default=0)
    p.add_argument("--trial_sample_neg_count", type=int, default=0)
    p.add_argument("--trial_sample_seed", type=int, default=42)

    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--no_use_cache", action="store_false", dest="use_cache")
    p.set_defaults(use_cache=True)
    p.add_argument("--cache_dir", type=str, default="", help="Default: <output_json_parent>/emb_cache_coded_only")
    p.add_argument("--overwrite_cache", action="store_true")
    p.add_argument("--cache_save_every", type=int, default=500)
    p.add_argument("--cache_incremental", action="store_true")
    p.add_argument("--no_cache_incremental", action="store_false", dest="cache_incremental")
    p.set_defaults(cache_incremental=True)

    p.add_argument("--plain_loader_batch_size", type=int, default=64)
    p.add_argument("--plain_num_workers", type=int, default=4)
    p.add_argument("--plain_backend_batch_size", type=int, default=32)
    p.add_argument("--plain_camp_batch_size", dest="plain_backend_batch_size", type=int, help="Backward-compatible alias for --plain_backend_batch_size.")

    p.add_argument("--dump_speaker_npy_dir", type=str, default="")
    p.add_argument("--speaker_id_sep", type=str, default="/")
    p.add_argument("--speaker_id_field", type=int, default=0)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--plot_title_fontsize", type=int, default=18)
    p.add_argument("--plot_label_fontsize", type=int, default=15)
    p.add_argument("--plot_tick_fontsize", type=int, default=13)
    p.add_argument("--plot_legend_fontsize", type=int, default=13)
    return p.parse_args()


def maybe_plot_summary(results: list[dict], output_png: Path, args: argparse.Namespace) -> None:
    if not results:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not found, skip png plotting. Install matplotlib to enable plotting.")
        return

    conditions = [str(r.get("condition", "")) for r in results]
    eers = [float(r.get("eer_percent", 0.0)) for r in results]
    min_dcfs = [float(r.get("min_dcf", 0.0)) for r in results]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.0, 5.0), squeeze=False)
    ax0 = axes[0][0]
    ax1 = axes[0][1]
    xs = list(range(len(conditions)))

    ax0.bar(xs, eers, color="#1f77b4", alpha=0.9, label="EER(%)")
    ax0.set_xticks(xs)
    ax0.set_xticklabels(conditions, rotation=20, ha="right")
    ax0.set_ylabel("EER (%)", fontsize=args.plot_label_fontsize)
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="best", fontsize=args.plot_legend_fontsize)
    ax0.tick_params(axis="both", labelsize=args.plot_tick_fontsize)

    ax1.bar(xs, min_dcfs, color="#d62728", alpha=0.9, label="minDCF")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(conditions, rotation=20, ha="right")
    ax1.set_ylabel("minDCF", fontsize=args.plot_label_fontsize)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=args.plot_legend_fontsize)
    ax1.tick_params(axis="both", labelsize=args.plot_tick_fontsize)

    fig.suptitle("Coded-Only Multi-Codec SV Metrics", fontsize=args.plot_title_fontsize)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    print(f"Saved plot: {output_png}")


def main() -> None:
    args = parse_args()
    if not args.backend_ckpt:
        args.backend_ckpt = args.campplus_ckpt
    if not args.backend_ckpt:
        raise ValueError("Please provide --backend_ckpt, or use the legacy --campplus_ckpt alias.")
    if args.trial_sample_fraction <= 0:
        raise ValueError("--trial_sample_fraction must be > 0")

    codec_scp_entries = parse_codec_scp_entries(args.codec_scp)
    all_trials = load_trials(args.trials_file)
    trials, trial_meta = stratified_sample_trials(
        all_trials,
        sample_fraction=args.trial_sample_fraction,
        sample_total=args.trial_sample_total,
        sample_pos_count=args.trial_sample_pos_count,
        sample_neg_count=args.trial_sample_neg_count,
        sample_seed=args.trial_sample_seed,
    )
    if not trials:
        raise RuntimeError("No trials available after stratified sampling.")

    required_utts = {u for _, u1, u2 in trials for u in (u1, u2)}

    import torch

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backend = build_backend(args.backend_type, args.backend_ckpt)
    backend.model.to(device)

    cache_dir = None
    backend_tag = Path(args.backend_ckpt).stem
    sample_tag = build_trial_sampling_cache_tag(args, trial_meta)
    if args.use_cache:
        cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else Path(args.output_json).resolve().parent / "emb_cache_coded_only"
        cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for codec_tag, scp_path in codec_scp_entries:
        scp_map = load_scp(scp_path)
        scp_eval = {utt: wav for utt, wav in scp_map.items() if utt in required_utts}
        missing = len(required_utts) - len(scp_eval)
        if missing > 0:
            print(f"[{codec_tag}] warning: {missing} utts in sampled trials are missing in scp")

        cache_path = None
        if cache_dir is not None:
            scp_tag = Path(scp_path).stem
            cache_path = cache_dir / f"codedonly__{codec_tag}__{scp_tag}__backend_{args.backend_type}_{backend_tag}{sample_tag}.npz"

        emb = extract_emb_plain(
            utt2wav=scp_eval,
            mode=f"coded_{codec_tag}",
            backend=backend,
            device=device,
            cache_path=cache_path,
            overwrite_cache=args.overwrite_cache,
            cache_save_every=args.cache_save_every,
            cache_incremental=args.cache_incremental,
            plain_loader_batch_size=args.plain_loader_batch_size,
            plain_num_workers=args.plain_num_workers,
            plain_backend_batch_size=args.plain_backend_batch_size,
        )

        labels, scores = score_trials(trials, emb)
        eer, min_dcf = compute_eer_mindcf(labels, scores)
        row = {
            "condition": f"coded_{codec_tag}",
            "num_trials": len(labels),
            "eer_percent": eer,
            "min_dcf": min_dcf,
            "missing_utts": missing,
        }
        results.append(row)
        print(f"[{codec_tag}] trials={len(labels)} eer={eer:.4f} minDCF={min_dcf:.6f}")

        if args.dump_speaker_npy_dir:
            dump_speaker_embeddings(
                emb,
                mode=f"coded_{codec_tag}",
                dump_root=Path(args.dump_speaker_npy_dir).resolve(),
                speaker_id_sep=args.speaker_id_sep,
                speaker_id_field=args.speaker_id_field,
            )

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")

    output_png = Path(args.output_png).resolve() if args.output_png else out_path.with_suffix(".png")
    maybe_plot_summary(results, output_png, args)


if __name__ == "__main__":
    main()
