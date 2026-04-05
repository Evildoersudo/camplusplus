#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\]\[(?P<phase>[^\]]+)\]\s+step\s+(?P<step>\d+)/(?P<total>\d+)\s+avg_train=(?P<avg>[0-9eE+\-.]+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot avg_train curve from train.log with step subsampling.")
    p.add_argument("--log_file", type=str, required=True, help="Path to train.log")
    p.add_argument("--sample_every", type=int, default=40, help="Keep one point every N parsed steps")
    p.add_argument("--output_png", type=str, default="", help="Output png path (default: alongside log)")
    p.add_argument("--title", type=str, default="avg_train Curve", help="Figure title")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file).resolve()
    if not log_file.is_file():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    sample_every = max(1, int(args.sample_every))
    output_png = Path(args.output_png).resolve() if args.output_png else (log_file.parent / "avg_train_curve.png")

    xs: list[int] = []
    ys: list[float] = []
    labels: list[str] = []

    parsed_count = 0
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            parsed_count += 1
            if parsed_count % sample_every != 0:
                continue
            xs.append(parsed_count)
            ys.append(float(m.group("avg")))
            labels.append(f"e{int(m.group('epoch'))}-{m.group('phase')}")

    if not xs:
        raise RuntimeError(
            "No sampled step points found. "
            "Check log format or lower --sample_every (e.g. 1/10)."
        )

    plt.figure(figsize=(10, 4.8))
    plt.plot(xs, ys, linewidth=1.5)
    plt.xlabel("Parsed step index")
    plt.ylabel("avg_train")
    plt.title(args.title + f" (sample_every={sample_every})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=160)
    print(f"saved plot: {output_png}")
    print(f"parsed step lines: {parsed_count}, plotted points: {len(xs)}")


if __name__ == "__main__":
    main()
