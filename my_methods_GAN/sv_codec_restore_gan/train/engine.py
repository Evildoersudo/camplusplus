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


def _frame_l1_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    t = min(a.shape[1], b.shape[1])
    return F.l1_loss(a[:, :t], b[:, :t])


def _build_loader(
    manifest: str,
    batch_size: int,
    num_workers: int,
    segment_seconds: float,
    train: bool,
    sample_fraction: float,
    sample_seed: int,
    stratified_sample: bool,
) -> DataLoader:
    ds = SVCodecPairDataset(
        manifest_csv=manifest,
        segment_seconds=segment_seconds,
        random_crop=train,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        stratified_sample=stratified_sample,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, collate_fn=collate_pair_batch)


def _save_ckpt(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _phase_from_epoch(epoch: int, p1: int, p2: int, p3: int) -> str:
    if epoch <= p1:
        return "phase1"
    if epoch <= p1 + p2:
        return "phase2"
    if epoch <= p1 + p2 + p3:
        return "phase3"
    return "done"


def _phase_segment_seconds(phase: str, args: argparse.Namespace) -> float:
    if phase == "phase2" and args.phase2_segment_seconds > 0:
        return float(args.phase2_segment_seconds)
    if phase == "phase3" and args.phase3_segment_seconds > 0:
        return float(args.phase3_segment_seconds)
    return float(args.segment_seconds)


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
    args.d_update_interval = max(1, int(args.d_update_interval))
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)

    train_loader = _build_loader(
        args.train_manifest,
        args.batch_size,
        args.num_workers,
        args.segment_seconds,
        True,
        args.train_sample_fraction,
        args.train_sample_seed,
        args.train_stratified_sample,
    )
    valid_loader = _build_loader(
        args.valid_manifest,
        args.batch_size,
        args.num_workers,
        args.segment_seconds,
        False,
        args.valid_sample_fraction,
        args.valid_sample_seed,
        args.valid_stratified_sample,
    )
    print(
        "[data] train samples={} valid samples={} (fractions: train={}, valid={})".format(
            len(train_loader.dataset),
            len(valid_loader.dataset),
            args.train_sample_fraction,
            args.valid_sample_fraction,
        )
    )

    generator = SVCodecRestoreGenerator(
        emb_dim=args.emb_dim,
        num_blocks=args.num_blocks,
        hidden_units=args.hidden_units,
        attn_heads=args.attn_heads,
    ).to(device)
    mrd = MultiResolutionDiscriminator().to(device) if args.phase3_use_gan else None
    mbd = MultiBandDiscriminator().to(device) if args.phase3_use_gan and args.use_mbd else None

    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr_g_max, weight_decay=args.weight_decay)
    disc_params = []
    if mrd is not None:
        disc_params.extend(mrd.parameters())
    if mbd is not None:
        disc_params.extend(mbd.parameters())
    opt_d = torch.optim.AdamW(disc_params, lr=args.lr_d_max, weight_decay=args.weight_decay) if disc_params else None

    phase1_epochs = int(args.phase1_epochs)
    phase2_epochs = int(args.phase2_epochs)
    phase3_epochs = int(args.phase3_epochs)
    total_epochs = phase1_epochs + phase2_epochs + phase3_epochs
    wavlm = None
    camp = None

    best_rec = float("inf")
    best_sv = float("-inf")
    history = []

    global_step = 0
    d_global_step = 0
    start_epoch = 1

    if getattr(args, "resume", False):
        if getattr(args, "resume_ckpt", ""):
            ckpt_path = Path(args.resume_ckpt).resolve()
        else:
            ckpt_path = _latest_checkpoint(out_dir / "checkpoints")

        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"Resume requested but no checkpoint found. resume_ckpt={getattr(args, 'resume_ckpt', '')}"
            )

        state = torch.load(str(ckpt_path), map_location="cpu")

        if not getattr(args, "resume_use_current_phase_config", False):
            prev_args = state.get("args", {}) if isinstance(state, dict) else {}
            if isinstance(prev_args, dict):
                phase1_epochs = int(prev_args.get("phase1_epochs", phase1_epochs))
                phase2_epochs = int(prev_args.get("phase2_epochs", phase2_epochs))
                phase3_epochs = int(prev_args.get("phase3_epochs", phase3_epochs))
                total_epochs = phase1_epochs + phase2_epochs + phase3_epochs
                print(
                    "[resume] keep checkpoint phase config: phase1={}, phase2={}, phase3={}".format(
                        phase1_epochs, phase2_epochs, phase3_epochs
                    )
                )
        else:
            print(
                "[resume] use current phase config: phase1={}, phase2={}, phase3={}".format(
                    phase1_epochs, phase2_epochs, phase3_epochs
                )
            )

        generator.load_state_dict(state["generator"])
        if mrd is not None and state.get("mrd") is not None:
            mrd.load_state_dict(state["mrd"])
        if mbd is not None and state.get("mbd") is not None:
            mbd.load_state_dict(state["mbd"])
        if "opt_g" in state:
            opt_g.load_state_dict(state["opt_g"])
        if opt_d is not None and state.get("opt_d") is not None:
            opt_d.load_state_dict(state["opt_d"])

        global_step = int(state.get("global_step", 0))
        d_global_step = int(state.get("d_global_step", 0))
        best_rec = float(state.get("best_valid_rec", state.get("best_valid", best_rec)))
        best_sv = float(state.get("best_valid_sv", best_sv))
        resume_epoch = int(state.get("epoch", 0))
        start_epoch = resume_epoch + 1
        print(f"[resume] loaded checkpoint: {ckpt_path}")
        print(f"[resume] start_epoch={start_epoch}, global_step={global_step}, d_global_step={d_global_step}")
    if start_epoch > total_epochs:
        print(
            f"[resume] start_epoch ({start_epoch}) > total_epochs ({total_epochs}). Nothing to train."
        )
        return

    phase_g_step = 0
    phase_d_step = 0
    prev_phase = ""
    phase_total_g_steps = max(1, len(train_loader))
    phase_total_d_steps = max(1, len(train_loader) // args.d_update_interval)

    for epoch in range(start_epoch, total_epochs + 1):
        phase = _phase_from_epoch(epoch, phase1_epochs, phase2_epochs, phase3_epochs)
        if phase == "done":
            break

        seg_seconds = _phase_segment_seconds(phase, args)
        train_loader = _build_loader(
            args.train_manifest,
            args.batch_size,
            args.num_workers,
            seg_seconds,
            True,
            args.train_sample_fraction,
            args.train_sample_seed,
            args.train_stratified_sample,
        )
        valid_loader = _build_loader(
            args.valid_manifest,
            args.batch_size,
            args.num_workers,
            seg_seconds,
            False,
            args.valid_sample_fraction,
            args.valid_sample_seed,
            args.valid_stratified_sample,
        )

        if phase != prev_phase:
            phase_g_step = 0
            phase_d_step = 0
            phase_epochs = {"phase1": phase1_epochs, "phase2": phase2_epochs, "phase3": phase3_epochs}[phase]
            phase_total_g_steps = max(1, phase_epochs * len(train_loader))
            phase_total_d_steps = max(1, (phase_epochs * len(train_loader) + args.d_update_interval - 1) // args.d_update_interval)
            prev_phase = phase

        need_camp = args.use_campplus_train_loss or args.valid_sv_metric
        if need_camp and camp is None:
            print("[info] Initializing CAM++ for validation/optional loss...")
            camp = FrozenCampPlus(args.campplus_ckpt).to(device)
            print("[info] CAM++ ready.")

        if phase == "phase3" and args.phase3_use_wavlm and wavlm is None:
            print("[info] Initializing local WavLM for frame-level distillation...")
            wavlm = WavLMFeatureExtractor(args.wavlm_root, args.wavlm_ckpt).to(device)
            print("[info] WavLM ready.")

        generator.train()
        if mrd is not None:
            mrd.train()
        if mbd is not None:
            mbd.train()

        train_loss = 0.0
        t0 = time.time()
        log_interval = max(1, len(train_loader) // 20)
        for step, batch in enumerate(train_loader, start=1):
            g_lr = _warmup_cosine_lr(
                step=phase_g_step,
                total_steps=phase_total_g_steps,
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

            rec_dict = loss_rec(
                restored,
                clean,
                si_sdr_weight=args.si_sdr_weight,
                mrstft_weight=args.mrstft_weight,
                complex_weight=args.complex_weight,
            )
            rec_term = args.rec_loss_weight * rec_dict["total"]

            adv_g = torch.zeros((), device=device)
            feat_g = torch.zeros((), device=device)
            wavlm_term = torch.zeros((), device=device)
            spk_term = torch.zeros((), device=device)

            if phase == "phase3" and opt_d is not None:
                d_lr = _warmup_cosine_lr(
                    step=phase_d_step,
                    total_steps=phase_total_d_steps,
                    warmup_steps=args.warmup_steps_d,
                    max_lr=args.lr_d_max,
                    min_lr=args.lr_d_min,
                )
                for group in opt_d.param_groups:
                    group["lr"] = d_lr

                do_d_step = ((step - 1) % args.d_update_interval == 0)
                real_mrd = mrd(clean) if mrd is not None else []
                fake_mrd = mrd(restored.detach()) if mrd is not None else []
                real_mbd = mbd(clean) if mbd is not None else []
                fake_mbd = mbd(restored.detach()) if mbd is not None else []

                d_loss = torch.zeros((), device=device)
                if real_mrd and fake_mrd:
                    d_loss = d_loss + loss_adv_discriminator(real_mrd, fake_mrd)
                if real_mbd and fake_mbd:
                    d_loss = d_loss + loss_adv_discriminator(real_mbd, fake_mbd)

                if do_d_step:
                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    opt_d.step()
                    d_global_step += 1
                    phase_d_step += 1

                fake_mrd_g = mrd(restored) if mrd is not None else []
                fake_mbd_g = mbd(restored) if mbd is not None else []
                if fake_mrd_g:
                    adv_g = adv_g + loss_adv_generator(fake_mrd_g)
                    feat_g = feat_g + loss_feature_matching(real_mrd, fake_mrd_g)
                if fake_mbd_g:
                    adv_g = adv_g + loss_adv_generator(fake_mbd_g)
                    feat_g = feat_g + loss_feature_matching(real_mbd, fake_mbd_g)

            if args.use_campplus_train_loss and phase in {"phase2", "phase3"} and camp is not None:
                clean_fbank = _fbank_batch(clean)
                rest_fbank = _fbank_batch(restored)
                with torch.no_grad():
                    emb_clean = camp(clean_fbank)
                emb_rest = camp(rest_fbank)
                spk_term = _cosine_loss(emb_rest, emb_clean)

            if phase == "phase3" and args.phase3_use_wavlm and wavlm is not None:
                with torch.no_grad():
                    wavlm_clean = wavlm(clean)
                wavlm_rest = wavlm(restored)
                if wavlm_rest.dim() == 2:
                    wavlm_term = _cosine_loss(wavlm_rest, wavlm_clean)
                else:
                    wavlm_term = _frame_l1_loss(wavlm_rest, wavlm_clean)

            g_loss = rec_term
            if args.use_campplus_train_loss and phase in {"phase2", "phase3"} and camp is not None:
                g_loss = g_loss + args.spk_loss_weight * spk_term
            if phase == "phase3":
                if opt_d is not None:
                    g_loss = g_loss + args.adv_loss_weight * adv_g + args.fm_loss_weight * feat_g
                if args.phase3_use_wavlm and wavlm is not None:
                    g_loss = g_loss + args.wavlm_loss_weight * wavlm_term

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=args.grad_clip)
            opt_g.step()

            train_loss += float(g_loss.detach().cpu())
            global_step += 1
            phase_g_step += 1

            if step % log_interval == 0 or step == len(train_loader):
                avg_train = train_loss / step
                elapsed = time.time() - t0
                print(
                    f"[epoch {epoch:03d}][{phase}] step {step}/{len(train_loader)} "
                    f"avg_train={avg_train:.4f} lr_g={opt_g.param_groups[0]['lr']:.2e} "
                    f"elapsed={elapsed:.1f}s"
                )

        train_loss /= max(1, len(train_loader))

        generator.eval()
        valid_rec_loss = 0.0
        valid_sv_cos_sum = 0.0
        valid_sv_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                coded = batch["coded"].to(device)
                clean = batch["clean"].to(device)
                restored = generator(coded)
                min_len = min(restored.shape[-1], clean.shape[-1])
                restored = restored[:, :min_len]
                clean = clean[:, :min_len]
                rec_dict = loss_rec(
                    restored,
                    clean,
                    si_sdr_weight=args.si_sdr_weight,
                    mrstft_weight=args.mrstft_weight,
                    complex_weight=args.complex_weight,
                )
                valid_rec_loss += float((args.rec_loss_weight * rec_dict["total"]).detach().cpu())

                if args.valid_sv_metric and camp is not None:
                    clean_fbank = _fbank_batch(clean)
                    rest_fbank = _fbank_batch(restored)
                    emb_clean = camp(clean_fbank)
                    emb_rest = camp(rest_fbank)
                    cos = F.cosine_similarity(emb_rest, emb_clean, dim=-1).mean()
                    valid_sv_cos_sum += float(cos.detach().cpu())
                    valid_sv_batches += 1

        valid_rec_loss /= max(1, len(valid_loader))
        valid_sv_cos = None
        if valid_sv_batches > 0:
            valid_sv_cos = valid_sv_cos_sum / float(valid_sv_batches)

        ckpt = {
            "epoch": epoch,
            "phase": phase,
            "global_step": global_step,
            "d_global_step": d_global_step,
            "best_valid_rec": best_rec,
            "best_valid_sv": best_sv,
            "generator": generator.state_dict(),
            "mrd": mrd.state_dict() if mrd is not None else None,
            "mbd": mbd.state_dict() if mbd is not None else None,
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict() if opt_d is not None else None,
            "args": vars(args),
        }
        _save_ckpt(out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", ckpt)

        if valid_rec_loss < best_rec:
            best_rec = valid_rec_loss
            _save_ckpt(out_dir / "best_generator.pt", ckpt)

        if valid_sv_cos is not None and valid_sv_cos > best_sv:
            best_sv = valid_sv_cos
            _save_ckpt(out_dir / "best_generator_sv.pt", ckpt)

        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "valid_loss": valid_rec_loss,
            "valid_rec_loss": valid_rec_loss,
            "valid_sv_cos": valid_sv_cos,
            "seconds": time.time() - t0,
            "lr_g": opt_g.param_groups[0]["lr"],
            "lr_d": opt_d.param_groups[0]["lr"] if opt_d is not None else 0.0,
        }
        history.append(row)
        sv_msg = f" sv_cos={valid_sv_cos:.4f}" if valid_sv_cos is not None else ""
        print(
            f"[epoch {epoch:03d}] phase={phase} train={train_loss:.4f} valid_rec={valid_rec_loss:.4f}{sv_msg} "
            f"time={row['seconds']:.1f}s"
        )

    save_json(out_dir / "train_summary.json", {"best_valid_rec": best_rec, "best_valid_sv": best_sv, "history": history})
