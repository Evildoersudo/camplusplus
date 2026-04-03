from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.compliance.kaldi as Kaldi

from sv_codec_restore_gan.data.dataset import SVCodecPairDataset, collate_pair_batch
from sv_codec_restore_gan.models.campplus_wrapper import FrozenCampPlus
from sv_codec_restore_gan.models.discriminators import MultiBandDiscriminator, MultiResolutionDiscriminator
from sv_codec_restore_gan.models.generator import SVCodecRestoreGenerator
from sv_codec_restore_gan.models.losses import (
    loss_adv_discriminator,
    loss_adv_generator,
    loss_feature_matching,
    loss_rec,
)
from sv_codec_restore_gan.models.wavlm_wrapper import WavLMFeatureExtractor
from sv_codec_restore_gan.utils.io import ensure_dir, save_json
from sv_codec_restore_gan.utils.seed import set_seed


def _fbank_batch(wav: torch.Tensor, sample_rate: int = 16000, n_mels: int = 80) -> torch.Tensor:
    feats = []
    for i in range(wav.shape[0]):
        one = wav[i : i + 1]
        feat = Kaldi.fbank(one, num_mel_bins=n_mels, sample_frequency=sample_rate, dither=0.0)
        feat = feat - feat.mean(0, keepdim=True)
        feats.append(feat)
    max_t = max(x.shape[0] for x in feats)
    out = []
    for feat in feats:
        if feat.shape[0] < max_t:
            feat = F.pad(feat, (0, 0, 0, max_t - feat.shape[0]))
        out.append(feat)
    return torch.stack(out, dim=0)


def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()


def _build_loader(manifest: str, batch_size: int, num_workers: int, segment_seconds: float, train: bool) -> DataLoader:
    ds = SVCodecPairDataset(manifest_csv=manifest, segment_seconds=segment_seconds, random_crop=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, collate_fn=collate_pair_batch)


def _save_ckpt(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def _phase_from_epoch(epoch: int, p1: int, p2: int, p3: int) -> str:
    if epoch <= p1:
        return "phase1"
    if epoch <= p1 + p2:
        return "phase2"
    if epoch <= p1 + p2 + p3:
        return "phase3"
    return "done"


def _warmup_cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if total_steps <= 0:
        return max_lr

    step = max(0, min(step, total_steps))
    warmup_steps = max(0, min(warmup_steps, total_steps))

    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)

    if total_steps == warmup_steps:
        return min_lr

    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    return min_lr + (max_lr - min_lr) * cosine


def train_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)

    train_loader = _build_loader(args.train_manifest, args.batch_size, args.num_workers, args.segment_seconds, True)
    valid_loader = _build_loader(args.valid_manifest, args.batch_size, args.num_workers, args.segment_seconds, False)

    generator = SVCodecRestoreGenerator(
        emb_dim=args.emb_dim,
        num_blocks=args.num_blocks,
        hidden_units=args.hidden_units,
        attn_heads=args.attn_heads,
    ).to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    mbd = MultiBandDiscriminator().to(device)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr_g_max, weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(list(mrd.parameters()) + list(mbd.parameters()), lr=args.lr_d_max, weight_decay=args.weight_decay)

    total_epochs = args.phase1_epochs + args.phase2_epochs + args.phase3_epochs
    wavlm = None
    camp = None

    if args.phase3_epochs > 0:
        wavlm = WavLMFeatureExtractor(args.wavlm_root, args.wavlm_ckpt).to(device)
        camp = FrozenCampPlus(args.campplus_ckpt).to(device)

    best_val = float("inf")
    history = []

    global_step = 0
    d_global_step = 0
    total_g_steps = total_epochs * len(train_loader)
    total_d_steps = (args.phase2_epochs + args.phase3_epochs) * len(train_loader)
    for epoch in range(1, total_epochs + 1):
        phase = _phase_from_epoch(epoch, args.phase1_epochs, args.phase2_epochs, args.phase3_epochs)
        if phase == "done":
            break

        generator.train()
        mrd.train()
        mbd.train()

        train_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            g_lr = _warmup_cosine_lr(
                step=global_step,
                total_steps=total_g_steps,
                warmup_steps=args.warmup_steps_g,
                max_lr=args.lr_g_max,
                min_lr=args.lr_g_min,
            )
            for group in opt_g.param_groups:
                group["lr"] = g_lr

            coded = batch["coded"].to(device)
            clean = batch["clean"].to(device)

            restored = generator(coded)
            min_len = min(restored.shape[-1], clean.shape[-1])
            restored = restored[:, :min_len]
            clean = clean[:, :min_len]

            rec_dict = loss_rec(restored, clean)
            rec_term = 10.0 * rec_dict["total"]

            adv_g = torch.zeros((), device=device)
            feat_g = torch.zeros((), device=device)
            wavlm_term = torch.zeros((), device=device)
            spk_term = torch.zeros((), device=device)

            if phase in {"phase2", "phase3"}:
                d_lr = _warmup_cosine_lr(
                    step=d_global_step,
                    total_steps=total_d_steps,
                    warmup_steps=args.warmup_steps_d,
                    max_lr=args.lr_d_max,
                    min_lr=args.lr_d_min,
                )
                for group in opt_d.param_groups:
                    group["lr"] = d_lr

                real_mrd = mrd(clean)
                fake_mrd = mrd(restored.detach())
                real_mbd = mbd(clean)
                fake_mbd = mbd(restored.detach())
                d_loss = 0.5 * loss_adv_discriminator(real_mrd, fake_mrd) + 0.5 * loss_adv_discriminator(real_mbd, fake_mbd)

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                d_global_step += 1

                fake_mrd_g = mrd(restored)
                fake_mbd_g = mbd(restored)
                adv_g = 0.5 * loss_adv_generator(fake_mrd_g) + 0.5 * loss_adv_generator(fake_mbd_g)
                feat_g = 0.5 * loss_feature_matching(real_mrd, fake_mrd_g) + 0.5 * loss_feature_matching(real_mbd, fake_mbd_g)

            if phase == "phase3":
                with torch.no_grad():
                    wavlm_clean = wavlm(clean)
                wavlm_rest = wavlm(restored)
                wavlm_term = _cosine_loss(wavlm_rest, wavlm_clean)

                clean_fbank = _fbank_batch(clean)
                rest_fbank = _fbank_batch(restored)
                with torch.no_grad():
                    emb_clean = camp(clean_fbank)
                emb_rest = camp(rest_fbank)
                spk_term = _cosine_loss(emb_rest, emb_clean)

            g_loss = rec_term
            if phase in {"phase2", "phase3"}:
                g_loss = g_loss + adv_g + 0.2 * feat_g
            if phase == "phase3":
                g_loss = g_loss + 1.0 * wavlm_term + 3.0 * spk_term

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=args.grad_clip)
            opt_g.step()

            train_loss += float(g_loss.detach().cpu())
            global_step += 1

        train_loss /= max(1, len(train_loader))

        generator.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                coded = batch["coded"].to(device)
                clean = batch["clean"].to(device)
                restored = generator(coded)
                min_len = min(restored.shape[-1], clean.shape[-1])
                restored = restored[:, :min_len]
                clean = clean[:, :min_len]
                rec_dict = loss_rec(restored, clean)
                valid_loss += float((10.0 * rec_dict["total"]).detach().cpu())
        valid_loss /= max(1, len(valid_loader))

        ckpt = {
            "epoch": epoch,
            "phase": phase,
            "global_step": global_step,
            "generator": generator.state_dict(),
            "mrd": mrd.state_dict(),
            "mbd": mbd.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "args": vars(args),
        }
        _save_ckpt(out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", ckpt)

        if valid_loss < best_val:
            best_val = valid_loss
            _save_ckpt(out_dir / "best_generator.pt", ckpt)

        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "seconds": time.time() - t0,
            "lr_g": opt_g.param_groups[0]["lr"],
            "lr_d": opt_d.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] phase={phase} train={train_loss:.4f} valid={valid_loss:.4f} "
            f"time={row['seconds']:.1f}s"
        )

    save_json(out_dir / "train_summary.json", {"best_valid": best_val, "history": history})
