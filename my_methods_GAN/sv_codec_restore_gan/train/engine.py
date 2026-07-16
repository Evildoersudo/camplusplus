from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.compliance.kaldi as Kaldi

from sv_codec_restore_gan.data.dataset import SVCodecPairDataset, collate_pair_batch
from sv_codec_restore_gan.models.campplus_wrapper import FrozenCampPlus, infer_campplus_embedding_dim
from sv_codec_restore_gan.models.discriminators import MultiBandDiscriminator, MultiResolutionDiscriminator
from sv_codec_restore_gan.models.generator import SVCodecRestoreGenerator
from sv_codec_restore_gan.models.losses import (
    loss_adv_discriminator,
    loss_adv_generator,
    loss_feature_matching,
    loss_rec,
)
from sv_codec_restore_gan.models.speaker_losses import AMSoftmaxClassifier
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


def _diff_mel_batch(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    win_length: int = 400,
    hop_length: int = 160,
    f_min: float = 20.0,
    f_max: float = 7600.0,
) -> torch.Tensor:
    # Differentiable log-mel frontend for CAMP++ teacher guidance.
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="slaney",
    ).to(wav.device)
    mel = mel_tf(wav)
    logmel = torch.log(mel.clamp_min(1e-6)).transpose(1, 2)
    return logmel - logmel.mean(dim=1, keepdim=True)


def _campplus_feats(wav: torch.Tensor, frontend: str) -> torch.Tensor:
    if frontend == "diff_mel":
        return _diff_mel_batch(wav)
    return _fbank_batch(wav)


def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()


def _frame_l1_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    t = min(a.shape[1], b.shape[1])
    return F.l1_loss(a[:, :t], b[:, :t])


def _campplus_feature_l1_loss(feats_a: dict[str, torch.Tensor], feats_b: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    losses = []
    common = sorted(set(feats_a.keys()) & set(feats_b.keys()))
    for k in common:
        a = feats_a[k]
        b = feats_b[k]
        if a.shape != b.shape:
            if a.dim() == b.dim() and a.dim() >= 3:
                t = min(a.shape[-1], b.shape[-1])
                a = a[..., :t]
                b = b[..., :t]
            else:
                continue
        losses.append(F.l1_loss(a, b))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def _build_loader(
    manifest: str,
    batch_size: int,
    num_workers: int,
    segment_seconds: float,
    train: bool,
    sample_fraction: float,
    sample_seed: int,
    stratified_sample: bool,
    codec_shift_samples: dict[str, int] | None,
    clean_passthrough_ratio: float = 0.0,
) -> DataLoader:
    ds = SVCodecPairDataset(
        manifest_csv=manifest,
        segment_seconds=segment_seconds,
        random_crop=train,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        stratified_sample=stratified_sample,
        codec_shift_samples=codec_shift_samples,
        clean_passthrough_ratio=clean_passthrough_ratio,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, collate_fn=collate_pair_batch)


def _format_sample_type_counts(loader: DataLoader) -> str:
    counts = {}
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "sample_type_counts"):
        counts = dataset.sample_type_counts()
    if not counts:
        return "unknown"
    return ",".join(f"{key}={counts[key]}" for key in sorted(counts))


def _zero_rec_dict(device: torch.device) -> dict[str, torch.Tensor]:
    z = torch.zeros((), device=device)
    return {
        "total": z,
        "si_sdr": z,
        "mrstft": z,
        "complex": z,
        "codec_total": z,
        "clean_total": z,
        "codec_count": z,
        "clean_count": z,
    }


def _optional_weight(args: argparse.Namespace, name: str, fallback_name: str) -> float:
    value = getattr(args, name, None)
    if value is None:
        return float(getattr(args, fallback_name))
    return float(value)


def _uses_split_clean_rec(args: argparse.Namespace, batch: dict) -> bool:
    is_clean = batch.get("is_clean")
    if is_clean is None or not bool(is_clean.any()):
        return False
    if abs(float(getattr(args, "clean_rec_loss_weight", 1.0)) - 1.0) > 1e-12:
        return True
    return any(getattr(args, name, None) is not None for name in (
        "clean_si_sdr_weight",
        "clean_mrstft_weight",
        "clean_complex_weight",
    ))


def _loss_rec_train(restored: torch.Tensor, clean: torch.Tensor, batch: dict, args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    if not _uses_split_clean_rec(args, batch):
        rec = loss_rec(
            restored,
            clean,
            si_sdr_weight=args.si_sdr_weight,
            mrstft_weight=args.mrstft_weight,
            complex_weight=args.complex_weight,
        )
        zero = torch.zeros((), device=device)
        return {
            **rec,
            "codec_total": rec["total"],
            "clean_total": zero,
            "codec_count": torch.tensor(float(restored.shape[0]), device=device),
            "clean_count": zero,
        }

    is_clean = batch["is_clean"].to(device=device, dtype=torch.bool)
    codec_mask = ~is_clean
    clean_mask = is_clean
    out = _zero_rec_dict(device)

    if bool(codec_mask.any()):
        codec_rec = loss_rec(
            restored[codec_mask],
            clean[codec_mask],
            si_sdr_weight=args.si_sdr_weight,
            mrstft_weight=args.mrstft_weight,
            complex_weight=args.complex_weight,
        )
        out["total"] = out["total"] + codec_rec["total"]
        out["si_sdr"] = out["si_sdr"] + codec_rec["si_sdr"]
        out["mrstft"] = out["mrstft"] + codec_rec["mrstft"]
        out["complex"] = out["complex"] + codec_rec["complex"]
        out["codec_total"] = codec_rec["total"]
        out["codec_count"] = codec_mask.sum().float()

    if bool(clean_mask.any()):
        clean_rec = loss_rec(
            restored[clean_mask],
            clean[clean_mask],
            si_sdr_weight=_optional_weight(args, "clean_si_sdr_weight", "si_sdr_weight"),
            mrstft_weight=_optional_weight(args, "clean_mrstft_weight", "mrstft_weight"),
            complex_weight=_optional_weight(args, "clean_complex_weight", "complex_weight"),
        )
        clean_weight = float(getattr(args, "clean_rec_loss_weight", 1.0))
        out["total"] = out["total"] + clean_weight * clean_rec["total"]
        out["clean_total"] = clean_rec["total"]
        out["clean_count"] = clean_mask.sum().float()
        if not bool(codec_mask.any()):
            out["si_sdr"] = clean_rec["si_sdr"]
            out["mrstft"] = clean_rec["mrstft"]
            out["complex"] = clean_rec["complex"]

    return out


def _grad_total_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros(())
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(g, ord=norm_type).to(device) for g in grads])
    return torch.linalg.vector_norm(norms, ord=norm_type)


def _load_generator_state(generator: torch.nn.Module, gen_state: dict, strict: bool, emit, label: str) -> None:
    result = generator.load_state_dict(gen_state, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing:
        emit(f"[{label}] generator missing keys: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if unexpected:
        emit(f"[{label}] generator unexpected keys: {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}")


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
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return min_lr + (max_lr - min_lr) * cosine


def _maybe_override_arch_from_init_ckpt(args: argparse.Namespace, emit) -> None:
    init_ckpt = getattr(args, "init_generator_ckpt", "")
    if getattr(args, "resume", False) or not init_ckpt:
        return
    init_path = Path(init_ckpt).resolve()
    if not init_path.exists():
        return
    state = torch.load(str(init_path), map_location="cpu")
    if not isinstance(state, dict):
        return
    ckpt_args = state.get("args", {})
    if not isinstance(ckpt_args, dict):
        return

    fields = ("emb_dim", "num_blocks", "hidden_units", "attn_heads")
    updates = []
    for field in fields:
        if field in ckpt_args:
            old_v = getattr(args, field)
            new_v = int(ckpt_args[field])
            if int(old_v) != new_v:
                setattr(args, field, new_v)
                updates.append((field, old_v, new_v))

    if updates:
        msg = ", ".join([f"{k}: {ov} -> {nv}" for k, ov, nv in updates])
        emit(f"[init] override generator arch from init ckpt args: {msg}")


def _build_speaker_index(rows) -> dict[str, int]:
    spk_set = sorted({row.spk_id for row in rows})
    return {spk: idx for idx, spk in enumerate(spk_set)}


def train_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.d_update_interval = max(1, int(args.d_update_interval))
    args.debug_campplus_grad_interval = max(1, int(getattr(args, "debug_campplus_grad_interval", 50)))
    args.campplus_feat_layers = tuple(x.strip() for x in str(getattr(args, "campplus_feat_layers", "")).split(",") if x.strip())
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.output_dir)
    log_file = (out_dir / "train.log").open("a", encoding="utf-8")

    def _emit(message: str, also_console: bool = True) -> None:
        if also_console:
            print(message)
        log_file.write(message + "\n")
        log_file.flush()

    codec_shift_samples: dict[str, int] = {}
    if bool(getattr(args, "enable_codec_time_align", False)):
        amr_shift = int(getattr(args, "amrwb_shift_samples", 0))
        if amr_shift != 0:
            codec_shift_samples["amrwb"] = amr_shift

    train_loader = _build_loader(
        args.train_manifest,
        args.batch_size,
        args.num_workers,
        args.segment_seconds,
        True,
        args.train_sample_fraction,
        args.train_sample_seed,
        args.train_stratified_sample,
        codec_shift_samples,
        float(getattr(args, "train_clean_passthrough_ratio", 0.0)),
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
        codec_shift_samples,
        float(getattr(args, "valid_clean_passthrough_ratio", 0.0)),
    )
    _emit(
        "[data] train samples={} valid samples={} (fractions: train={}, valid={})".format(
            len(train_loader.dataset),
            len(valid_loader.dataset),
            args.train_sample_fraction,
            args.valid_sample_fraction,
        )
    )
    _emit(
        "[data] sample_types train=[{}] valid=[{}] clean_passthrough_ratio train={} valid={}".format(
            _format_sample_type_counts(train_loader),
            _format_sample_type_counts(valid_loader),
            float(getattr(args, "train_clean_passthrough_ratio", 0.0)),
            float(getattr(args, "valid_clean_passthrough_ratio", 0.0)),
        )
    )
    _emit(f"[data] codec_time_align={bool(getattr(args, 'enable_codec_time_align', False))} shifts={codec_shift_samples}")

    _maybe_override_arch_from_init_ckpt(args, _emit)

    spk_to_idx: dict[str, int] = {}
    spk_classifier = None
    if args.use_spk_amsoftmax:
        train_rows = getattr(train_loader.dataset, "rows", [])
        spk_to_idx = _build_speaker_index(train_rows)
        if len(spk_to_idx) <= 1:
            raise RuntimeError("AM-Softmax requires at least 2 speakers in train manifest.")
        spk_emb_dim = infer_campplus_embedding_dim(args.campplus_ckpt)
        spk_classifier = AMSoftmaxClassifier(
            in_dim=spk_emb_dim,
            num_classes=len(spk_to_idx),
            margin=args.spk_am_margin,
            scale=args.spk_am_scale,
        ).to(device)
        _emit(f"[info] AM-Softmax enabled: speakers={len(spk_to_idx)} emb_dim={spk_emb_dim}")

    generator = SVCodecRestoreGenerator(
        emb_dim=args.emb_dim,
        num_blocks=args.num_blocks,
        hidden_units=args.hidden_units,
        attn_heads=args.attn_heads,
        cws_mode=getattr(args, "cws_mode", "uniform"),
        cws_band_edges=getattr(args, "cws_band_edges", ""),
    ).to(device)
    mrd = MultiResolutionDiscriminator().to(device) if args.phase3_use_gan else None
    mbd = MultiBandDiscriminator().to(device) if args.phase3_use_gan and args.use_mbd else None

    g_params = list(generator.parameters())
    if spk_classifier is not None:
        g_params.extend(spk_classifier.parameters())
    opt_g = torch.optim.AdamW(g_params, lr=args.lr_g_max, weight_decay=args.weight_decay)
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

    if total_epochs <= 0:
        raise ValueError("Total epochs must be > 0. Check phase1/phase2/phase3 epoch settings.")

    # Enforce warm-start for phase3-only setup to match Experiment C design.
    if phase1_epochs == 0 and phase2_epochs == 0 and phase3_epochs > 0:
        if not getattr(args, "resume", False) and not getattr(args, "init_generator_ckpt", ""):
            raise ValueError(
                "Phase3-only training requires --init_generator_ckpt (or --resume) to warm-start from Experiment B checkpoint."
            )

    _emit(
        "[config] phases: phase1={} phase2={} phase3={} | phase3_use_gan={} use_mbd={} phase3_use_wavlm={} | "
        "spk_loss_weight={} adv_loss_weight={} fm_loss_weight={}".format(
            phase1_epochs,
            phase2_epochs,
            phase3_epochs,
            args.phase3_use_gan,
            args.use_mbd,
            args.phase3_use_wavlm,
            args.spk_loss_weight,
            args.adv_loss_weight,
            args.fm_loss_weight,
        )
    )
    _emit(
        "[config] cws_mode={} cws_band_edges={}".format(
            getattr(args, "cws_mode", "uniform"),
            getattr(args, "cws_band_edges", "") or "default",
        )
    )
    _emit(
        "[config] clean_passthrough: train_ratio={} valid_ratio={} clean_rec_weight={} "
        "clean_si_sdr_weight={} clean_mrstft_weight={} clean_complex_weight={} codec_only_sv_loss={}".format(
            float(getattr(args, "train_clean_passthrough_ratio", 0.0)),
            float(getattr(args, "valid_clean_passthrough_ratio", 0.0)),
            float(getattr(args, "clean_rec_loss_weight", 1.0)),
            getattr(args, "clean_si_sdr_weight", None),
            getattr(args, "clean_mrstft_weight", None),
            getattr(args, "clean_complex_weight", None),
            bool(getattr(args, "codec_only_sv_loss", False)),
        )
    )
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
                _emit(
                    "[resume] keep checkpoint phase config: phase1={}, phase2={}, phase3={}".format(
                        phase1_epochs, phase2_epochs, phase3_epochs
                    )
                )
        else:
            _emit(
                "[resume] use current phase config: phase1={}, phase2={}, phase3={}".format(
                    phase1_epochs, phase2_epochs, phase3_epochs
                )
            )

        _load_generator_state(
            generator,
            state["generator"],
            strict=False,
            emit=_emit,
            label="resume",
        )
        if mrd is not None and state.get("mrd") is not None:
            mrd.load_state_dict(state["mrd"])
        if mbd is not None and state.get("mbd") is not None:
            mbd.load_state_dict(state["mbd"])
        if spk_classifier is not None and state.get("spk_classifier") is not None:
            spk_classifier.load_state_dict(state["spk_classifier"])
        if "opt_g" in state:
            try:
                opt_g.load_state_dict(state["opt_g"])
            except ValueError as exc:
                _emit(f"[resume] skip opt_g state due param-group mismatch: {exc}")
        if opt_d is not None and state.get("opt_d") is not None:
            opt_d.load_state_dict(state["opt_d"])

        global_step = int(state.get("global_step", 0))
        d_global_step = int(state.get("d_global_step", 0))
        best_rec = float(state.get("best_valid_rec", state.get("best_valid", best_rec)))
        best_sv = float(state.get("best_valid_sv", best_sv))
        resume_epoch = int(state.get("epoch", 0))
        start_epoch = resume_epoch + 1
        _emit(f"[resume] loaded checkpoint: {ckpt_path}")
        _emit(f"[resume] start_epoch={start_epoch}, global_step={global_step}, d_global_step={d_global_step}")
    elif getattr(args, "init_generator_ckpt", ""):
        init_path = Path(args.init_generator_ckpt).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"init_generator_ckpt not found: {init_path}")
        init_state = torch.load(str(init_path), map_location="cpu")
        gen_state = init_state["generator"] if isinstance(init_state, dict) and "generator" in init_state else init_state
        _load_generator_state(
            generator,
            gen_state,
            strict=False,
            emit=_emit,
            label="init",
        )
        _emit(f"[init] loaded generator warm-start from: {init_path}")
    if start_epoch > total_epochs:
        _emit(
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
            codec_shift_samples,
            float(getattr(args, "train_clean_passthrough_ratio", 0.0)),
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
            codec_shift_samples,
            float(getattr(args, "valid_clean_passthrough_ratio", 0.0)),
        )

        if phase != prev_phase:
            phase_g_step = 0
            phase_d_step = 0
            phase_epochs = {"phase1": phase1_epochs, "phase2": phase2_epochs, "phase3": phase3_epochs}[phase]
            phase_total_g_steps = max(1, phase_epochs * len(train_loader))
            phase_total_d_steps = max(1, (phase_epochs * len(train_loader) + args.d_update_interval - 1) // args.d_update_interval)
            prev_phase = phase

        need_camp = args.use_campplus_train_loss or args.valid_sv_metric or (spk_classifier is not None) or args.use_campplus_feat_loss
        if need_camp and camp is None:
            _emit("[info] Initializing CAM++ for validation/optional loss...")
            camp = FrozenCampPlus(args.campplus_ckpt).to(device)
            _emit("[info] CAM++ ready.")

        if phase == "phase3" and args.phase3_use_wavlm and wavlm is None:
            _emit("[info] Initializing local WavLM for frame-level distillation...")
            wavlm = WavLMFeatureExtractor(args.wavlm_root, args.wavlm_ckpt).to(device)
            _emit("[info] WavLM ready.")

        generator.train()
        if mrd is not None:
            mrd.train()
        if mbd is not None:
            mbd.train()

        train_loss = 0.0
        sum_sisdr = 0.0
        sum_mrstft = 0.0
        sum_complex = 0.0
        t0 = time.time()
        log_interval = max(1, int(getattr(args, "log_interval", 50)))
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

            rec_dict = _loss_rec_train(restored, clean, batch, args, device)
            rec_term = args.rec_loss_weight * rec_dict["total"]

            # 分项累计
            sum_sisdr += float(rec_dict["si_sdr"].detach().cpu())
            sum_mrstft += float(rec_dict["mrstft"].detach().cpu())
            sum_complex += float(rec_dict["complex"].detach().cpu())

            adv_g = torch.zeros((), device=device)
            feat_g = torch.zeros((), device=device)
            d_loss = torch.zeros((), device=device)
            wavlm_term = torch.zeros((), device=device)
            spk_term = torch.zeros((), device=device)
            spk_weighted_term = torch.zeros((), device=device)
            spk_cls_term = torch.zeros((), device=device)
            spk_cls_weighted_term = torch.zeros((), device=device)
            spk_feat_term = torch.zeros((), device=device)
            spk_feat_weighted_term = torch.zeros((), device=device)
            gan_adv_weighted_term = torch.zeros((), device=device)
            gan_fm_weighted_term = torch.zeros((), device=device)
            d_lr_current = 0.0
            camp_grad_msg = ""

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
                d_lr_current = float(d_lr)

                do_d_step = ((step - 1) % args.d_update_interval == 0)
                real_mrd = mrd(clean) if mrd is not None else []
                fake_mrd = mrd(restored.detach()) if mrd is not None else []
                real_mbd = mbd(clean) if mbd is not None else []
                fake_mbd = mbd(restored.detach()) if mbd is not None else []

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

            sv_clean = clean
            sv_restored = restored
            sv_spk_ids = batch["spk_id"]
            if bool(getattr(args, "codec_only_sv_loss", False)) and "is_clean" in batch:
                codec_mask_for_sv = (~batch["is_clean"].to(device=device, dtype=torch.bool))
                if bool(codec_mask_for_sv.any()):
                    sv_clean = clean[codec_mask_for_sv]
                    sv_restored = restored[codec_mask_for_sv]
                    keep_flags = codec_mask_for_sv.detach().cpu().tolist()
                    sv_spk_ids = [spk for spk, keep in zip(batch["spk_id"], keep_flags) if keep]
                else:
                    sv_clean = clean[:0]
                    sv_restored = restored[:0]
                    sv_spk_ids = []

            if phase in {"phase2", "phase3"} and camp is not None and sv_clean.shape[0] > 0 and (
                args.use_campplus_train_loss or args.use_campplus_feat_loss or spk_classifier is not None
            ):
                clean_fbank = _campplus_feats(sv_clean, args.campplus_frontend)
                rest_fbank = _campplus_feats(sv_restored, args.campplus_frontend)
                need_feats = bool(args.use_campplus_feat_loss and args.campplus_feat_layers)

                if need_feats:
                    emb_rest, feat_rest = camp(rest_fbank, return_feats=True, feat_layers=args.campplus_feat_layers)
                else:
                    emb_rest = camp(rest_fbank)
                    feat_rest = {}

                emb_clean = None
                feat_clean = {}
                if args.use_campplus_train_loss or args.use_campplus_feat_loss:
                    with torch.no_grad():
                        if need_feats:
                            emb_clean, feat_clean = camp(clean_fbank, return_feats=True, feat_layers=args.campplus_feat_layers)
                        else:
                            emb_clean = camp(clean_fbank)

                if args.use_campplus_train_loss and emb_clean is not None:
                    spk_term = _cosine_loss(emb_rest, emb_clean)

                if args.use_campplus_feat_loss and feat_rest and feat_clean:
                    spk_feat_term = _campplus_feature_l1_loss(feat_rest, feat_clean, device=device)

                if args.use_campplus_train_loss and args.debug_campplus_grad and (step == 1 or step % args.debug_campplus_grad_interval == 0):
                    if not spk_term.requires_grad:
                        camp_grad_msg = "camp_grad=disconnected"
                    else:
                        grad_from_camp = torch.autograd.grad(
                            outputs=args.spk_loss_weight * spk_term,
                            inputs=restored,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad_from_camp is None:
                            camp_grad_msg = "camp_grad=none"
                        else:
                            camp_grad_msg = (
                                "camp_grad_norm={:.3e} camp_grad_abs_mean={:.3e} camp_grad_abs_max={:.3e}".format(
                                    float(grad_from_camp.norm().detach().cpu()),
                                    float(grad_from_camp.abs().mean().detach().cpu()),
                                    float(grad_from_camp.abs().max().detach().cpu()),
                                )
                            )

                if spk_classifier is not None:
                    labels = torch.tensor([spk_to_idx[s] for s in sv_spk_ids], dtype=torch.long, device=device)
                    spk_cls_term = spk_classifier(emb_rest, labels)

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
                spk_weighted_term = args.spk_loss_weight * spk_term
                g_loss = g_loss + spk_weighted_term
            if args.use_campplus_feat_loss and phase in {"phase2", "phase3"} and camp is not None:
                spk_feat_weighted_term = args.campplus_feat_loss_weight * spk_feat_term
                g_loss = g_loss + spk_feat_weighted_term
            if spk_classifier is not None and phase in {"phase2", "phase3"}:
                spk_cls_weighted_term = args.spk_cls_loss_weight * spk_cls_term
                g_loss = g_loss + spk_cls_weighted_term
            if phase == "phase3":
                if opt_d is not None:
                    gan_adv_weighted_term = args.adv_loss_weight * adv_g
                    gan_fm_weighted_term = args.fm_loss_weight * feat_g
                    g_loss = g_loss + gan_adv_weighted_term + gan_fm_weighted_term
                if args.phase3_use_wavlm and wavlm is not None:
                    g_loss = g_loss + args.wavlm_loss_weight * wavlm_term

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            grad_norm_pre_clip = torch.nn.utils.clip_grad_norm_(g_params, max_norm=args.grad_clip)
            grad_norm_post_clip = _grad_total_norm(g_params)
            opt_g.step()

            train_loss += float(g_loss.detach().cpu())
            global_step += 1
            phase_g_step += 1

            should_log = step == 1 or step == len(train_loader) or (step % log_interval == 0)
            if not should_log:
                continue

            avg_train = train_loss / step
            elapsed = time.time() - t0
            rec_total = float(rec_term.detach().cpu())
            spk_raw = float(spk_term.detach().cpu())
            spk_weighted = float(spk_weighted_term.detach().cpu())
            spk_cls_raw = float(spk_cls_term.detach().cpu())
            spk_cls_weighted = float(spk_cls_weighted_term.detach().cpu())
            spk_feat_raw = float(spk_feat_term.detach().cpu())
            spk_feat_weighted = float(spk_feat_weighted_term.detach().cpu())
            gan_d_raw = float(d_loss.detach().cpu())
            gan_adv_raw = float(adv_g.detach().cpu())
            gan_fm_raw = float(feat_g.detach().cpu())
            gan_adv_weighted = float(gan_adv_weighted_term.detach().cpu())
            gan_fm_weighted = float(gan_fm_weighted_term.detach().cpu())
            generator_grad_norm_pre_clip = (
                float(grad_norm_pre_clip.detach().cpu()) if torch.is_tensor(grad_norm_pre_clip) else float(grad_norm_pre_clip)
            )
            generator_grad_norm_post_clip = (
                float(grad_norm_post_clip.detach().cpu()) if torch.is_tensor(grad_norm_post_clip) else float(grad_norm_post_clip)
            )
            rec_codec_total = float(rec_dict.get("codec_total", torch.zeros((), device=device)).detach().cpu())
            rec_clean_total = float(rec_dict.get("clean_total", torch.zeros((), device=device)).detach().cpu())
            codec_batch = int(float(rec_dict.get("codec_count", torch.zeros((), device=device)).detach().cpu()))
            clean_batch = int(float(rec_dict.get("clean_count", torch.zeros((), device=device)).detach().cpu()))
            step_message = (
                f"[epoch {epoch:03d}][{phase}] step {step}/{len(train_loader)} "
                f"avg_train={avg_train:.4f} "
                f"rec_total={rec_total:.4f} "
                f"rec_codec_total={rec_codec_total:.4f} "
                f"rec_clean_total={rec_clean_total:.4f} "
                f"codec_batch={codec_batch} "
                f"clean_batch={clean_batch} "
                f"spk_raw={spk_raw:.4f} "
                f"spk_weighted={spk_weighted:.4f} "
                f"spk_feat_raw={spk_feat_raw:.4f} "
                f"spk_feat_weighted={spk_feat_weighted:.4f} "
                f"spk_cls_raw={spk_cls_raw:.4f} "
                f"spk_cls_weighted={spk_cls_weighted:.4f} "
                f"gan_d_raw={gan_d_raw:.4f} "
                f"gan_adv_raw={gan_adv_raw:.4f} "
                f"gan_fm_raw={gan_fm_raw:.4f} "
                f"gan_adv_weighted={gan_adv_weighted:.4f} "
                f"gan_fm_weighted={gan_fm_weighted:.4f} "
                f"generator_grad_norm_pre_clip={generator_grad_norm_pre_clip:.3e} "
                f"generator_grad_norm_post_clip={generator_grad_norm_post_clip:.3e} "
                f"si_sdr={sum_sisdr/step:.4f} "
                f"mrstft={sum_mrstft/step:.4f} "
                f"complex={sum_complex/step:.4f} "
                f"lr_g={opt_g.param_groups[0]['lr']:.2e} "
                f"lr_d={d_lr_current:.2e} "
                f"elapsed={elapsed:.1f}s"
            )
            if camp_grad_msg:
                step_message = f"{step_message} {camp_grad_msg}"
            _emit(step_message, also_console=True)

        train_loss /= max(1, len(train_loader))

        generator.eval()
        valid_rec_loss = 0.0
        valid_sv_cos_sum = 0.0
        valid_sv_batches = 0
        with torch.no_grad():
            for valid_step, batch in enumerate(valid_loader, start=1):
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
                    clean_fbank = _campplus_feats(clean, args.campplus_frontend)
                    rest_fbank = _campplus_feats(restored, args.campplus_frontend)
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
            "spk_classifier": spk_classifier.state_dict() if spk_classifier is not None else None,
            "spk_to_idx": spk_to_idx,
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
        _emit(
            f"[epoch {epoch:03d}] phase={phase} train={train_loss:.4f} valid_rec={valid_rec_loss:.4f}{sv_msg} "
            f"time={row['seconds']:.1f}s"
        )

    save_json(out_dir / "train_summary.json", {"best_valid_rec": best_rec, "best_valid_sv": best_sv, "history": history})
