#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


STEP_LINE_RE = re.compile(r"\[epoch\s+(?P<epoch>\d+)\]\[(?P<phase>[^\]]+)\]\s+step\s+")
KV_RE = re.compile(r"(?P<k>[A-Za-z_][A-Za-z0-9_]*)=(?P<v>[0-9eE+\-.]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot epoch-level mean/std trends for speaker losses from train.log")
    p.add_argument("--log_file", type=str, default="", help="Path to train.log")
    p.add_argument("--phase", type=str, default="phase2", help="Phase name to include (e.g. phase2)")
    p.add_argument(
        "--metrics",
        type=str,
        default="spk_raw,spk_feat_raw",
        help="Comma-separated metrics parsed from step lines",
    )
    p.add_argument("--output_png", type=str, default="", help="Output png path")
    p.add_argument("--output_csv", type=str, default="", help="Output csv summary path")
    p.add_argument("--title", type=str, default="Epoch Speaker Loss Trend", help="Plot title")
    p.add_argument(
        "--plot_report_trend",
        action="store_true",
        help="Plot report trend from train_summary.json + epoch_phase2_spk_trend.csv.",
    )
    p.add_argument(
        "--summary_json",
        type=str,
        default="",
        help="Path to train_summary.json (used when --plot_report_trend is enabled).",
    )
    p.add_argument(
        "--trend_csv",
        type=str,
        default="",
        help="Path to epoch_phase2_spk_trend.csv (used when --plot_report_trend is enabled).",
    )
    p.add_argument(
        "--report_output_csv",
        type=str,
        default="",
        help="Optional merged csv path for report trend mode.",
    )
    return p.parse_args()


def _calc_mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / float(n)
    var = sum((x - mean) * (x - mean) for x in values) / float(n)
    return mean, math.sqrt(max(var, 0.0))


def _load_valid_sv_cos(summary_json: Path) -> dict[int, tuple[float, float]]:
    with summary_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    history = payload.get("history", [])
    out: dict[int, tuple[float, float]] = {}
    for row in history:
        if not isinstance(row, dict):
            continue
        if "epoch" not in row or "valid_sv_cos" not in row:
            continue
        epoch = int(row["epoch"])
        mean = float(row["valid_sv_cos"])
        # train_summary.json usually has epoch-level scalar valid_sv_cos only.
        # If *_std key exists, use it; otherwise fallback to 0 for plotting.
        std = float(
            row.get(
                "valid_sv_cos_std",
                row.get("valid_sv_std", row.get("sv_cos_std", 0.0)),
            )
        )
        out[epoch] = (mean, std)
    return out


def _load_spk_means(trend_csv: Path) -> dict[int, tuple[float, float, float, float]]:
    out: dict[int, tuple[float, float, float, float]] = {}
    with trend_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            spk_raw_mean = float(row["spk_raw_mean"])
            spk_raw_std = float(row.get("spk_raw_std", 0.0))
            spk_feat_raw_mean = float(row["spk_feat_raw_mean"])
            spk_feat_raw_std = float(row.get("spk_feat_raw_std", 0.0))
            out[epoch] = (spk_raw_mean, spk_raw_std, spk_feat_raw_mean, spk_feat_raw_std)
    return out


def run_report_trend_mode(args: argparse.Namespace) -> None:
    summary_json = Path(args.summary_json).resolve() if args.summary_json else Path(
        "my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/train_summary.json"
    ).resolve()
    trend_csv = Path(args.trend_csv).resolve() if args.trend_csv else Path(
        "my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_spk_trend.csv"
    ).resolve()

    if not summary_json.is_file():
        raise FileNotFoundError(f"summary_json not found: {summary_json}")
    if not trend_csv.is_file():
        raise FileNotFoundError(f"trend_csv not found: {trend_csv}")

    valid_sv_by_epoch = _load_valid_sv_cos(summary_json)
    spk_means_by_epoch = _load_spk_means(trend_csv)
    if not valid_sv_by_epoch:
        raise RuntimeError(f"No valid_sv_cos found in: {summary_json}")
    if not spk_means_by_epoch:
        raise RuntimeError(f"No speaker means found in: {trend_csv}")

    epochs = sorted(set(valid_sv_by_epoch.keys()) & set(spk_means_by_epoch.keys()))
    if not epochs:
        raise RuntimeError("No overlapped epochs between summary_json and trend_csv")

    merged_rows: list[dict[str, float | int]] = []
    valid_sv_has_nonzero_std = False
    for epoch in epochs:
        valid_sv_mean, valid_sv_std = valid_sv_by_epoch[epoch]
        spk_raw_mean, spk_raw_std, spk_feat_raw_mean, spk_feat_raw_std = spk_means_by_epoch[epoch]
        valid_sv_has_nonzero_std = valid_sv_has_nonzero_std or (valid_sv_std > 0.0)
        merged_rows.append(
            {
                "epoch": epoch,
                "valid_sv_cos": valid_sv_mean,
                "valid_sv_cos_std": valid_sv_std,
                "spk_raw_mean": spk_raw_mean,
                "spk_raw_std": spk_raw_std,
                "spk_feat_raw_mean": spk_feat_raw_mean,
                "spk_feat_raw_std": spk_feat_raw_std,
            }
        )

    output_png = Path(args.output_png).resolve() if args.output_png else (trend_csv.parent / "epoch_phase2_validsv_spk_trend.png")
    merged_csv = Path(args.report_output_csv).resolve() if args.report_output_csv else (trend_csv.parent / "epoch_phase2_validsv_spk_trend.csv")

    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    with merged_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "valid_sv_cos",
                "valid_sv_cos_std",
                "spk_raw_mean",
                "spk_raw_std",
                "spk_feat_raw_mean",
                "spk_feat_raw_std",
            ],
        )
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)
    print(f"saved merged csv: {merged_csv}")
    if not valid_sv_has_nonzero_std:
        print("warning: valid_sv_cos std not found in summary_json; using 0.0 as fallback")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not found, skip png plotting. Install matplotlib to enable plotting.")
        return

    valid_sv = [float(r["valid_sv_cos"]) for r in merged_rows]
    valid_sv_std = [float(r["valid_sv_cos_std"]) for r in merged_rows]
    spk_raw = [float(r["spk_raw_mean"]) for r in merged_rows]
    spk_raw_std = [float(r["spk_raw_std"]) for r in merged_rows]
    spk_feat = [float(r["spk_feat_raw_mean"]) for r in merged_rows]
    spk_feat_std = [float(r["spk_feat_raw_std"]) for r in merged_rows]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.5, 9.0), sharex=True, squeeze=False)
    ax0 = axes[0][0]
    ax1 = axes[1][0]
    ax2 = axes[2][0]

    ax0.plot(epochs, valid_sv, marker="o", linewidth=1.8, color="#1f77b4", label="valid_sv_cos mean")
    ax0.fill_between(
        epochs,
        [m - s for m, s in zip(valid_sv, valid_sv_std)],
        [m + s for m, s in zip(valid_sv, valid_sv_std)],
        alpha=0.20,
        color="#1f77b4",
        label="mean±std",
    )
    ax0.set_ylabel("valid_sv_cos")
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="best")

    ax1.plot(epochs, spk_raw, marker="o", linewidth=1.6, color="#d62728", label="spk_raw_mean")
    ax1.fill_between(
        epochs,
        [m - s for m, s in zip(spk_raw, spk_raw_std)],
        [m + s for m, s in zip(spk_raw, spk_raw_std)],
        alpha=0.20,
        color="#d62728",
        label="mean±std",
    )
    ax1.set_ylabel("spk_raw")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best")

    ax2.plot(epochs, spk_feat, marker="o", linewidth=1.6, color="#2ca02c", label="spk_feat_raw_mean")
    ax2.fill_between(
        epochs,
        [m - s for m, s in zip(spk_feat, spk_feat_std)],
        [m + s for m, s in zip(spk_feat, spk_feat_std)],
        alpha=0.20,
        color="#2ca02c",
        label="mean±std",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("spk_feat_raw")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best")

    title = args.title if args.title else "Validation SV and Speaker Loss Trend"
    fig.suptitle(title)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    print(f"saved plot: {output_png}")


def main() -> None:
    args = parse_args()

    if args.plot_report_trend:
        run_report_trend_mode(args)
        return

    if not args.log_file:
        raise ValueError("--log_file is required when --plot_report_trend is not enabled")

    log_file = Path(args.log_file).resolve()
    if not log_file.is_file():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("--metrics is empty")

    per_epoch: dict[int, dict[str, list[float]]] = {}
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = STEP_LINE_RE.search(line)
            if not m:
                continue
            if m.group("phase") != args.phase:
                continue

            epoch = int(m.group("epoch"))
            kvs = {k: float(v) for k, v in KV_RE.findall(line)}
            if any(metric not in kvs for metric in metrics):
                continue

            slot = per_epoch.setdefault(epoch, {metric: [] for metric in metrics})
            for metric in metrics:
                slot[metric].append(kvs[metric])

    if not per_epoch:
        raise RuntimeError(f"No step lines found for phase '{args.phase}' with metrics: {','.join(metrics)}")

    epochs = sorted(per_epoch.keys())
    rows: list[dict[str, float | int]] = []
    for epoch in epochs:
        row: dict[str, float | int] = {"epoch": epoch, "count": len(per_epoch[epoch][metrics[0]])}
        for metric in metrics:
            mean, std = _calc_mean_std(per_epoch[epoch][metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_min"] = min(per_epoch[epoch][metric])
            row[f"{metric}_max"] = max(per_epoch[epoch][metric])
        rows.append(row)

    output_png = Path(args.output_png).resolve() if args.output_png else (log_file.parent / f"epoch_{args.phase}_spk_trend.png")
    output_csv = Path(args.output_csv).resolve() if args.output_csv else (log_file.parent / f"epoch_{args.phase}_spk_trend.csv")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "count"]
    for metric in metrics:
        fieldnames.extend([
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_min",
            f"{metric}_max",
        ])
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"saved csv: {output_csv}")
    print(f"epochs: {len(epochs)}, metrics: {','.join(metrics)}")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not found, skip png plotting. Install matplotlib to enable --output_png.")
        return

    n = len(metrics)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8.0, 3.0 * n), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for i, metric in enumerate(metrics):
        ax = flat_axes[i]
        ys = [float(r[f"{metric}_mean"]) for r in rows]
        ystd = [float(r[f"{metric}_std"]) for r in rows]
        ylow = [y - s for y, s in zip(ys, ystd)]
        yhigh = [y + s for y, s in zip(ys, ystd)]

        ax.plot(epochs, ys, marker="o", linewidth=1.4, label=f"{metric} mean")
        ax.fill_between(epochs, ylow, yhigh, alpha=0.20, label="mean±std")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

    fig.suptitle(f"{args.title} ({args.phase})")
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=160)
    print(f"saved plot: {output_png}")


if __name__ == "__main__":
    main()
