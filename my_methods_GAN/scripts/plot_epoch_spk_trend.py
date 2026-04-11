#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


STEP_LINE_RE = re.compile(r"\[epoch\s+(?P<epoch>\d+)\]\[(?P<phase>[^\]]+)\]\s+step\s+")
KV_RE = re.compile(r"(?P<k>[A-Za-z_][A-Za-z0-9_]*)=(?P<v>[0-9eE+\-.]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot epoch-level mean/std trends for speaker losses from train.log")
    p.add_argument("--log_file", type=str, required=True, help="Path to train.log")
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
    return p.parse_args()


def _calc_mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / float(n)
    var = sum((x - mean) * (x - mean) for x in values) / float(n)
    return mean, math.sqrt(max(var, 0.0))


def main() -> None:
    args = parse_args()
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
