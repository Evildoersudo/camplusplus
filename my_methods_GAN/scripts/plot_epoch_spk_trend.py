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
EPOCH_SUMMARY_RE = re.compile(r"\[epoch\s+(?P<epoch>\d+)\]\s+phase=(?P<phase>\S+)")

TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13


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
    p.add_argument(
        "--epoch_range",
        type=str,
        default="",
        help="Epoch range to plot, e.g. 1-10 or 1,3,5-8. Empty means all.",
    )
    p.add_argument(
        "--valid_sv_source",
        type=str,
        default="summary_json",
        choices=["summary_json", "train_log"],
        help="Source of valid_sv_cos in report trend mode.",
    )
    p.add_argument(
        "--valid_sv_log_file",
        type=str,
        default="",
        help="Path to train.log for valid_sv_cos when --valid_sv_source=train_log.",
    )
    return p.parse_args()


def _calc_mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / float(n)
    var = sum((x - mean) * (x - mean) for x in values) / float(n)
    return mean, math.sqrt(max(var, 0.0))


def _parse_epoch_range(text: str) -> set[int] | None:
    s = text.strip()
    if not s:
        return None
    out: set[int] = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a = int(a_str.strip())
            b = int(b_str.strip())
            if a <= 0 or b <= 0:
                raise ValueError("epoch_range must be positive integers")
            lo, hi = (a, b) if a <= b else (b, a)
            out.update(range(lo, hi + 1))
        else:
            v = int(token)
            if v <= 0:
                raise ValueError("epoch_range must be positive integers")
            out.add(v)
    if not out:
        raise ValueError("epoch_range parsed as empty set")
    return out


def _apply_epoch_filter(epochs: list[int], epoch_filter: set[int] | None) -> list[int]:
    if epoch_filter is None:
        return epochs
    return [e for e in epochs if e in epoch_filter]


def _load_valid_metrics_from_summary(
    summary_json: Path,
    phase_filter: str = "",
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    with summary_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    history = payload.get("history", [])
    valid_sv_out: dict[int, tuple[float, float]] = {}
    valid_rec_out: dict[int, tuple[float, float]] = {}
    for row in history:
        if not isinstance(row, dict):
            continue
        if "epoch" not in row:
            continue
        row_phase = str(row.get("phase", "")).strip()
        if phase_filter and row_phase and row_phase != phase_filter:
            continue
        epoch = int(row["epoch"])
        if "valid_sv_cos" in row:
            sv_mean = float(row["valid_sv_cos"])
            # train_summary.json usually has epoch-level scalar valid_sv_cos only.
            # If *_std key exists, use it; otherwise fallback to 0 for plotting.
            sv_std = float(
                row.get(
                    "valid_sv_cos_std",
                    row.get("valid_sv_std", row.get("sv_cos_std", 0.0)),
                )
            )
            valid_sv_out[epoch] = (sv_mean, sv_std)
        if "valid_rec_loss" in row:
            rec_mean = float(row["valid_rec_loss"])
            rec_std = float(
                row.get(
                    "valid_rec_loss_std",
                    row.get("valid_rec_std", row.get("valid_loss_std", 0.0)),
                )
            )
            valid_rec_out[epoch] = (rec_mean, rec_std)
    return valid_sv_out, valid_rec_out


def _load_valid_sv_cos_from_log(log_file: Path, phase_filter: str = "") -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = EPOCH_SUMMARY_RE.search(line)
            if not m:
                continue
            phase = m.group("phase")
            if phase_filter and phase != phase_filter:
                continue
            kvs = {k: float(v) for k, v in KV_RE.findall(line)}
            if "sv_cos" not in kvs and "valid_sv_cos" not in kvs:
                continue
            epoch = int(m.group("epoch"))
            sv = float(kvs.get("sv_cos", kvs.get("valid_sv_cos", 0.0)))
            out[epoch] = (sv, 0.0)
    return out


def _load_valid_rec_loss_from_log(log_file: Path, phase_filter: str = "") -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = EPOCH_SUMMARY_RE.search(line)
            if not m:
                continue
            phase = m.group("phase")
            if phase_filter and phase != phase_filter:
                continue
            kvs = {k: float(v) for k, v in KV_RE.findall(line)}
            if "valid_rec" in kvs:
                rec = float(kvs["valid_rec"])
            elif "valid_rec_loss" in kvs:
                rec = float(kvs["valid_rec_loss"])
            elif "valid_loss" in kvs:
                rec = float(kvs["valid_loss"])
            else:
                continue
            epoch = int(m.group("epoch"))
            out[epoch] = (rec, 0.0)
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
    epoch_filter = _parse_epoch_range(args.epoch_range)
    summary_json = Path(args.summary_json).resolve() if args.summary_json else Path(
        "my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/train_summary.json"
    ).resolve()
    trend_csv = Path(args.trend_csv).resolve() if args.trend_csv else Path(
        "my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_spk_trend.csv"
    ).resolve()
    valid_sv_log = Path(args.valid_sv_log_file).resolve() if args.valid_sv_log_file else (trend_csv.parent / "train.log")

    if not trend_csv.is_file():
        raise FileNotFoundError(f"trend_csv not found: {trend_csv}")
    if args.valid_sv_source == "summary_json":
        if not summary_json.is_file():
            raise FileNotFoundError(f"summary_json not found: {summary_json}")
    else:
        if not valid_sv_log.is_file():
            raise FileNotFoundError(f"valid_sv_log_file not found: {valid_sv_log}")

    if args.valid_sv_source == "summary_json":
        valid_sv_by_epoch, valid_rec_by_epoch = _load_valid_metrics_from_summary(summary_json, phase_filter=args.phase)
    else:
        valid_sv_by_epoch = _load_valid_sv_cos_from_log(valid_sv_log, phase_filter=args.phase)
        valid_rec_by_epoch = _load_valid_rec_loss_from_log(valid_sv_log, phase_filter=args.phase)
    spk_means_by_epoch = _load_spk_means(trend_csv)
    if not valid_sv_by_epoch:
        if args.valid_sv_source == "summary_json":
            raise RuntimeError(f"No valid_sv_cos found in: {summary_json}")
        raise RuntimeError(f"No sv_cos found in: {valid_sv_log}")
    if not valid_rec_by_epoch:
        if args.valid_sv_source == "summary_json":
            raise RuntimeError(f"No valid_rec_loss found in: {summary_json}")
        raise RuntimeError(f"No valid_rec/valid_rec_loss found in: {valid_sv_log}")
    if not spk_means_by_epoch:
        raise RuntimeError(f"No speaker means found in: {trend_csv}")

    epochs = sorted(set(valid_sv_by_epoch.keys()) & set(valid_rec_by_epoch.keys()) & set(spk_means_by_epoch.keys()))
    epochs = _apply_epoch_filter(epochs, epoch_filter)
    if not epochs:
        raise RuntimeError("No overlapped epochs after applying source/range filter")

    merged_rows: list[dict[str, float | int]] = []
    valid_sv_has_nonzero_std = False
    for epoch in epochs:
        valid_rec_mean, valid_rec_std = valid_rec_by_epoch[epoch]
        valid_sv_mean, valid_sv_std = valid_sv_by_epoch[epoch]
        spk_raw_mean, spk_raw_std, spk_feat_raw_mean, spk_feat_raw_std = spk_means_by_epoch[epoch]
        valid_sv_has_nonzero_std = valid_sv_has_nonzero_std or (valid_sv_std > 0.0)
        merged_rows.append(
            {
                "epoch": epoch,
                "valid_rec_loss": valid_rec_mean,
                "valid_rec_loss_std": valid_rec_std,
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
                "valid_rec_loss",
                "valid_rec_loss_std",
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
        print("warning: valid_sv_cos std not found in source; using 0.0 as fallback")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not found, skip png plotting. Install matplotlib to enable plotting.")
        return

    valid_rec = [float(r["valid_rec_loss"]) for r in merged_rows]
    valid_sv = [float(r["valid_sv_cos"]) for r in merged_rows]
    valid_sv_std = [float(r["valid_sv_cos_std"]) for r in merged_rows]
    spk_raw = [float(r["spk_raw_mean"]) for r in merged_rows]
    spk_raw_std = [float(r["spk_raw_std"]) for r in merged_rows]
    spk_feat = [float(r["spk_feat_raw_mean"]) for r in merged_rows]
    spk_feat_std = [float(r["spk_feat_raw_std"]) for r in merged_rows]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.0, 8.0), sharex=True, squeeze=False)
    ax0 = axes[0][0]
    ax1 = axes[0][1]
    ax2 = axes[1][0]
    ax3 = axes[1][1]

    ax0.plot(epochs, valid_rec, marker="o", linewidth=1.8, color="#9467bd", label="valid_rec")
    ax0.set_ylabel("valid_rec", fontsize=LABEL_FONTSIZE)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax0.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    ax1.plot(epochs, valid_sv, marker="o", linewidth=1.8, color="#1f77b4", label="valid_sv_cos mean")
    ax1.fill_between(
        epochs,
        [m - s for m, s in zip(valid_sv, valid_sv_std)],
        [m + s for m, s in zip(valid_sv, valid_sv_std)],
        alpha=0.20,
        color="#1f77b4",
        label="mean±std",
    )
    ax1.set_ylabel("valid_sv_cos", fontsize=LABEL_FONTSIZE)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax1.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    ax2.plot(epochs, spk_raw, marker="o", linewidth=1.6, color="#d62728", label="spk_raw_mean")
    ax2.fill_between(
        epochs,
        [m - s for m, s in zip(spk_raw, spk_raw_std)],
        [m + s for m, s in zip(spk_raw, spk_raw_std)],
        alpha=0.20,
        color="#d62728",
        label="mean±std",
    )
    ax2.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("spk_raw", fontsize=LABEL_FONTSIZE)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax2.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    ax3.plot(epochs, spk_feat, marker="o", linewidth=1.6, color="#2ca02c", label="spk_feat_raw_mean")
    ax3.fill_between(
        epochs,
        [m - s for m, s in zip(spk_feat, spk_feat_std)],
        [m + s for m, s in zip(spk_feat, spk_feat_std)],
        alpha=0.20,
        color="#2ca02c",
        label="mean±std",
    )
    ax3.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel("spk_feat_raw", fontsize=LABEL_FONTSIZE)
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax3.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    title = args.title if args.title else "Validation SV and Speaker Loss Trend"
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180)
    print(f"saved plot: {output_png}")


def main() -> None:
    args = parse_args()
    epoch_filter = _parse_epoch_range(args.epoch_range)

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
    epochs = _apply_epoch_filter(epochs, epoch_filter)
    if not epochs:
        raise RuntimeError("No epochs left after applying --epoch_range filter")
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
        ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(metric, fontsize=LABEL_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.suptitle(f"{args.title} ({args.phase})", fontsize=TITLE_FONTSIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=160)
    print(f"saved plot: {output_png}")


if __name__ == "__main__":
    main()
