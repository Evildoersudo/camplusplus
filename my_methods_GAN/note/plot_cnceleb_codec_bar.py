#!/usr/bin/env python3
"""Plot CN-Celeb coded-only baseline bar chart for EER and minDCF."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    codecs = ["Clean", "Opus", "AMR-WB", "AAC", "G.711"]
    eer = [7.1360, 9.1692, 10.4703, 7.9401, 12.9019]
    mindcf = [0.4112, 0.5123, 0.5149, 0.4428, 0.6583]

    x = np.arange(len(codecs))
    bar_w = 0.38

    fig, ax1 = plt.subplots(figsize=(9.5, 5.4), dpi=160)

    bars1 = ax1.bar(x - bar_w / 2, eer, width=bar_w, color="#4C78A8", label="EER (%)")
    ax1.set_ylabel("EER (%)", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(codecs)
    ax1.set_xlabel("Codec")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + bar_w / 2, mindcf, width=bar_w, color="#F58518", label="minDCF")
    ax2.set_ylabel("minDCF", color="#F58518")
    ax2.tick_params(axis="y", labelcolor="#F58518")

    for b in bars1:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h + 0.12,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#2f4f6f",
        )
    for b in bars2:
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            h + 0.008,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#a65414",
        )

    ax1.legend([bars1, bars2], ["EER (%)", "minDCF"], loc="upper left", frameon=False)
    plt.title("CN-Celeb coded-only baseline: EER and minDCF by codec")
    plt.tight_layout()

    out_dir = Path("my_methods_GAN/note/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "cnceleb_codec_eer_mindcf_bar.png"
    out_svg = out_dir / "cnceleb_codec_eer_mindcf_bar.svg"
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
