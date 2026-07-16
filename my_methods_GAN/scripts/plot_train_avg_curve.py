#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\]\[(?P<phase>[^\]]+)\]\s+step\s+(?P<step>\d+)/(?P<total>\d+)\s+avg_train=(?P<avg>[0-9eE+\-.]+)"
)
KV_RE = re.compile(r"(?P<k>[A-Za-z_][A-Za-z0-9_]*)=(?P<v>[0-9eE+\-.]+)")

DEFAULT_METRICS = [
    "avg_train",
    "si_sdr",
    "mrstft",
    "complex",
]

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training metric curves from train.log with step subsampling.")
    p.add_argument("--log_file", type=str, required=True, help="Path to train.log")
    p.add_argument("--sample_every", type=int, default=40, help="Keep one point every N parsed steps")
    p.add_argument("--output_png", type=str, default="", help="Output png path (default: alongside log)")
    p.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names to plot.",
    )
    p.add_argument("--title", type=str, default="Train Metrics Curves", help="Figure title")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file).resolve()
    if not log_file.is_file():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    sample_every = max(1, int(args.sample_every))
    output_png = Path(args.output_png).resolve() if args.output_png else (log_file.parent / "train_metrics_curve.png")
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("--metrics is empty.")

    xs: list[int] = []
    values: dict[str, list[float]] = {m: [] for m in metrics}

    parsed_count = 0
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            parsed_count += 1
            if parsed_count % sample_every != 0:
                continue
            kvs = {k: float(v) for k, v in KV_RE.findall(line)}
            if any(m not in kvs for m in metrics):
                continue
            xs.append(parsed_count)
            for metric in metrics:
                values[metric].append(kvs[metric])

    if not xs:
        raise RuntimeError(
            "No sampled step points found. "
            "Check log format or lower --sample_every (e.g. 1/10)."
        )

    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 3.0 * nrows), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for i, metric in enumerate(metrics):
        ax = flat_axes[i]
        ax.plot(xs, values[metric], linewidth=1.2)
        ax.set_xlabel("Parsed step index", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(metric, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.35)

    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(args.title + f" (sample_every={sample_every})", fontsize=TITLE_FONTSIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=160)
    print(f"saved plot: {output_png}")
    print(f"parsed step lines: {parsed_count}, plotted points: {len(xs)}, metrics: {','.join(metrics)}")


if __name__ == "__main__":
    main()
