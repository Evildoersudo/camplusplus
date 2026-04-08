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

    p.add_argument("--phase1_epochs", type=int, default=25)
    p.add_argument("--phase2_epochs", type=int, default=8)
    p.add_argument("--phase3_epochs", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--segment_seconds", type=float, default=2.0)
    p.add_argument("--phase2_segment_seconds", type=float, default=4.0)
    p.add_argument("--phase3_segment_seconds", type=float, default=4.0)
    p.add_argument("--train_sample_fraction", type=float, default=1.0, help="Fraction of train manifest rows to use.")
    p.add_argument("--valid_sample_fraction", type=float, default=1.0, help="Fraction of valid manifest rows to use.")
    p.add_argument("--train_sample_seed", type=int, default=42, help="Random seed for train subset sampling.")
    p.add_argument("--valid_sample_seed", type=int, default=43, help="Random seed for valid subset sampling.")
    p.add_argument("--train_stratified_sample", action="store_true", help="Use speaker-stratified sampling for train subset.")
    p.add_argument("--no_train_stratified_sample", action="store_false", dest="train_stratified_sample", help="Disable speaker-stratified sampling for train subset.")
    p.add_argument("--valid_stratified_sample", action="store_true", help="Use speaker-stratified sampling for valid subset.")
    p.add_argument("--no_valid_stratified_sample", action="store_false", dest="valid_stratified_sample", help="Disable speaker-stratified sampling for valid subset.")
    p.set_defaults(train_stratified_sample=True, valid_stratified_sample=True)

    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--attn_heads", type=int, default=4)

    p.add_argument("--lr_g_max", type=float, default=3e-4, help="Generator max learning rate for warmup-cosine schedule.")
    p.add_argument("--lr_g_min", type=float, default=1e-5, help="Generator minimum learning rate for cosine decay.")
    p.add_argument("--lr_d_max", type=float, default=5e-5, help="Discriminator max learning rate for warmup-cosine schedule.")
    p.add_argument("--lr_d_min", type=float, default=1e-6, help="Discriminator minimum learning rate for cosine decay.")
    p.add_argument("--warmup_steps_g", type=int, default=1000, help="Linear warmup steps for generator.")
    p.add_argument("--warmup_steps_d", type=int, default=1000, help="Linear warmup steps for discriminator.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)

    p.add_argument("--si_sdr_weight", type=float, default=1.0)
    p.add_argument("--mrstft_weight", type=float, default=1.0)
    p.add_argument("--complex_weight", type=float, default=0.5)
    p.add_argument("--rec_loss_weight", type=float, default=1.0)
    p.add_argument("--spk_loss_weight", type=float, default=0.1)
    p.add_argument("--wavlm_loss_weight", type=float, default=0.1)
    p.add_argument("--adv_loss_weight", type=float, default=0.1)
    p.add_argument("--fm_loss_weight", type=float, default=0.1)
    p.add_argument("--d_update_interval", type=int, default=2, help="Update discriminator every N generator steps.")

    p.add_argument("--use_campplus_train_loss", action="store_true", help="Use CAMP++ embedding loss during training.")
    p.add_argument("--no_use_campplus_train_loss", action="store_false", dest="use_campplus_train_loss", help="Disable CAMP++ training loss (recommended default).")
    p.add_argument(
        "--campplus_frontend",
        type=str,
        default="kaldi",
        choices=["kaldi", "diff_mel"],
        help="Frontend for CAMP++ train/valid features: kaldi (legacy) or diff_mel (differentiable mel).",
    )
    p.add_argument("--valid_sv_metric", action="store_true", help="Compute validation speaker cosine metric with CAMP++.")
    p.add_argument("--no_valid_sv_metric", action="store_false", dest="valid_sv_metric", help="Disable validation speaker cosine metric.")
    p.add_argument(
        "--debug_campplus_grad",
        action="store_true",
        help="Print gradient stats propagated from CAMP++ speaker loss back to restored waveform.",
    )
    p.add_argument(
        "--debug_campplus_grad_interval",
        type=int,
        default=50,
        help="Print CAMP++ gradient stats every N train steps when --debug_campplus_grad is enabled.",
    )
    p.set_defaults(use_campplus_train_loss=False, valid_sv_metric=True)

    p.add_argument("--phase3_use_gan", action="store_true", help="Enable adversarial training in phase3.")
    p.add_argument("--phase3_no_gan", action="store_false", dest="phase3_use_gan", help="Disable adversarial training in phase3.")
    p.add_argument("--use_mbd", action="store_true", help="Enable extra MBD discriminator in phase3 GAN training.")
    p.add_argument("--no_use_mbd", action="store_false", dest="use_mbd", help="Disable MBD and use MRD only.")
    p.add_argument("--phase3_use_wavlm", action="store_true", help="Enable frame-level WavLM distillation in phase3.")
    p.add_argument("--phase3_no_wavlm", action="store_false", dest="phase3_use_wavlm", help="Disable WavLM distillation in phase3.")
    p.set_defaults(phase3_use_gan=False, use_mbd=False, phase3_use_wavlm=False)

    p.add_argument("--wavlm_root", type=str, default="/home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM")
    p.add_argument("--wavlm_ckpt", type=str, default="/home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt")
    p.add_argument("--campplus_ckpt", type=str, default="/home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin")

    p.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
    p.add_argument("--resume_ckpt", type=str, default="", help="Optional checkpoint path. If empty, use latest under output_dir/checkpoints.")
    p.add_argument(
        "--init_generator_ckpt",
        type=str,
        default="",
        help="Optional warm-start checkpoint for generator weights only (e.g., Experiment A best_generator.pt).",
    )
    p.add_argument(
        "--resume_use_current_phase_config",
        action="store_true",
        help="When resuming, use current --phase*_epochs instead of checkpoint phase config.",
    )
    p.add_argument(
        "--resume_keep_phase_config",
        action="store_false",
        dest="resume_use_current_phase_config",
        help="When resuming, keep checkpoint phase config (default behavior).",
    )
    p.set_defaults(resume_use_current_phase_config=False)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    train_main(args)


if __name__ == "__main__":
    main()
