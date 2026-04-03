#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sv_codec_restore_gan.train.engine import train_main


def parse_args():
    p = argparse.ArgumentParser(description="Train SV-CodecRestoreGAN with 3-phase strategy.")
    p.add_argument("--train_manifest", type=str, required=True)
    p.add_argument("--valid_manifest", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--phase1_epochs", type=int, default=10)
    p.add_argument("--phase2_epochs", type=int, default=10)
    p.add_argument("--phase3_epochs", type=int, default=10)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--segment_seconds", type=float, default=2.0)

    p.add_argument("--emb_dim", type=int, default=48)
    p.add_argument("--num_blocks", type=int, default=5)
    p.add_argument("--hidden_units", type=int, default=100)
    p.add_argument("--attn_heads", type=int, default=4)

    p.add_argument("--lr_g_max", type=float, default=5e-4, help="Generator max learning rate for warmup-cosine schedule.")
    p.add_argument("--lr_g_min", type=float, default=5e-6, help="Generator minimum learning rate for cosine decay.")
    p.add_argument("--lr_d_max", type=float, default=1e-4, help="Discriminator max learning rate for warmup-cosine schedule.")
    p.add_argument("--lr_d_min", type=float, default=1e-6, help="Discriminator minimum learning rate for cosine decay.")
    p.add_argument("--warmup_steps_g", type=int, default=1000, help="Linear warmup steps for generator.")
    p.add_argument("--warmup_steps_d", type=int, default=1000, help="Linear warmup steps for discriminator.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)

    p.add_argument("--wavlm_root", type=str, default="/home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM")
    p.add_argument("--wavlm_ckpt", type=str, default="/home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM/WavLM-Base.pt")
    p.add_argument("--campplus_ckpt", type=str, default="/home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    train_main(args)


if __name__ == "__main__":
    main()
