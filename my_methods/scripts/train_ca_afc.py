"""Train the CA-AFC frontend with a frozen CAM++ backend.

Training follows the two-stage plan in `my_methods/note/method_way.md`:
1. Reconstruction pretraining.
2. Task-oriented tuning with frozen CAM++ embedding feedback.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_methods.data.ca_afc_data import (
    PairFeatureDataset,
    PrecomputedPairFeatureDataset,
    collate_pair_batch,
    cosine_embedding_consistency,
    latest_frontend_checkpoint,
    load_frozen_campplus,
    mask_from_lengths,
    save_json,
    smoothness_loss,
    weighted_reconstruction_loss,
)
from my_methods.models.ca_afc_frontend import CAAFCFrontend


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CA-AFC frontend with frozen CAM++ supervision.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Parameter notes for the common training command:\n"
            "  --train_feature_manifest  preferred offline feature manifest built by precompute_pair_features.py\n"
            "  --train_manifest   clean/codec pair list built by build_pair_manifest.py\n"
            "  --output_dir       directory used to save checkpoints and train_summary.json\n"
            "  --campplus_model_bin  pretrained CAM++ backend used only for frozen embedding supervision\n"
            "  --sample_rate      waveform sample rate used when loading audio\n"
            "  --max_frames       number of frames kept per training chunk; 300 is about a short utterance segment\n"
            "  --batch_size       number of pair samples per optimizer step\n"
            "  --pretrain_epochs  stage-1 epochs using only reconstruction + smoothness losses\n"
            "  --finetune_epochs  stage-2 epochs adding frozen CAM++ embedding consistency loss\n"
            "  --lambda_rec       weight for feature reconstruction loss\n"
            "  --lambda_emb       weight for CAM++ embedding consistency loss in stage 2\n"
            "  --lambda_smooth    weight for temporal smoothness regularization\n"
            "  --rec_loss_reduction frame keeps legacy scale; element also averages across 80 bins\n"
            "  --log_interval     print batch progress every N steps\n"
            "  --max_train_samples optional cap for quick smoke tests on the train split\n"
            "  --max_valid_samples optional cap for quick smoke tests on the validation split\n"
            "  --device           training device, usually cuda or cpu\n"
        ),
    )
    parser.add_argument(
        "--train_feature_manifest",
        type=str,
        default="",
        help="Preferred offline feature manifest CSV produced by precompute_pair_features.py. When set, training reads .pt tensors instead of wav files.",
    )
    parser.add_argument(
        "--valid_feature_manifest",
        type=str,
        default="",
        help="Optional offline validation feature manifest CSV. Used together with --train_feature_manifest.",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        default="",
        help="Raw pair manifest CSV for training. Used only when train_feature_manifest is not provided.",
    )
    parser.add_argument(
        "--valid_manifest",
        type=str,
        default="",
        help="Optional raw validation pair manifest CSV. Used only when valid_feature_manifest is not provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Training output directory. Saves best_frontend.pt, checkpoints/*.pt, and train_summary.json here.",
    )
    parser.add_argument(
        "--campplus_model_bin",
        type=str,
        required=True,
        help="Pretrained CAM++ checkpoint used as the frozen backend in stage-2 embedding supervision.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Waveform sample rate used when loading clean/codec audio. Should match your dataset and CAM++ setup.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Training chunk length in frames after crop/pad. 300 means each sample is trimmed/padded to 300 frames.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of paired utterances per optimization step. Larger values use more GPU memory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker count. Increase this when disk I/O becomes the bottleneck.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden channel size of the CA-AFC frontend branches and fusion layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout ratio used inside the CA-AFC fusion representation.",
    )
    parser.add_argument(
        "--pretrain_dropout",
        type=float,
        default=0.0,
        help="Dropout ratio used in stage-1 pretraining. Setting this to 0 can improve early stability.",
    )
    parser.add_argument(
        "--band_scale",
        type=float,
        default=1.0,
        help="Band reweighting range controller. 1.0 means frame-wise band weights in approximately [0, 2].",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=0.1,
        help="Residual head scaling factor. Smaller values reduce residual branch dominance in early training.",
    )
    parser.add_argument(
        "--codec_vocab",
        type=str,
        default="clean,aac,opus,amrwb,g711,unknown",
        help="Comma-separated codec vocabulary used by codec-conditioned frontend, e.g. clean,aac,opus,amrwb,unknown.",
    )
    parser.add_argument(
        "--codec_emb_dim",
        type=int,
        default=16,
        help="Codec embedding size used by the codec-conditioned modulation block.",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=5,
        help="Stage-1 epoch count. Only reconstruction loss and smoothness loss are optimized in this stage.",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=10,
        help="Stage-2 epoch count. Reconstruction loss is kept and frozen CAM++ embedding consistency is added.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Base learning rate for the selected optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay coefficient used to regularize CA-AFC parameters.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer type. adamw is the default; sgd enables momentum SGD training.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum used when --optimizer sgd is selected.",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="Enable Nesterov momentum when using SGD.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "step"],
        help="Learning-rate schedule. cosine uses per-step warmup + cosine decay; step uses epoch decay.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of optimizer steps used for linear learning-rate warmup.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate reached by cosine decay.",
    )
    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        default=0.1,
        help="Decay factor used by the step scheduler.",
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=int,
        default=0,
        help="Step scheduler period in epochs. 0 disables epoch decay even if --scheduler step is selected.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=0.0,
        help="Optional gradient clipping max norm. 0 disables clipping.",
    )
    parser.add_argument(
        "--lambda_rec",
        type=float,
        default=1.0,
        help="Weight of the weighted FBank reconstruction loss. This is the main supervision term in both stages.",
    )
    parser.add_argument(
        "--lambda_emb",
        type=float,
        default=0.3,
        help="Weight of the frozen CAM++ embedding consistency loss. Used only in stage 2.",
    )
    parser.add_argument(
        "--emb_term_target_ratio",
        type=float,
        default=0.1,
        help="Target ratio between emb term and rec term in finetune. 0 disables adaptive emb scaling.",
    )
    parser.add_argument(
        "--emb_lambda_scale_min",
        type=float,
        default=1.0,
        help="Lower bound for adaptive emb-loss scaling factor.",
    )
    parser.add_argument(
        "--emb_lambda_scale_max",
        type=float,
        default=20.0,
        help="Upper bound for adaptive emb-loss scaling factor.",
    )
    parser.add_argument(
        "--lambda_smooth",
        type=float,
        default=0.01,
        help="Weight of the temporal smoothness regularizer applied to the predicted residual.",
    )
    parser.add_argument(
        "--rec_loss_reduction",
        type=str,
        default="element",
        choices=["frame", "element"],
        help="Reconstruction loss normalization mode. `frame` matches legacy behavior; `element` also averages over 80 bins.",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio when valid_manifest is not provided. Example: 0.1 means 10%% validation.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Optional train-set cap for smoke tests. 0 means use the full training split.",
    )
    parser.add_argument(
        "--max_valid_samples",
        type=int,
        default=0,
        help="Optional validation-set cap for smoke tests. 0 means use the full validation split.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Print progress every N batches inside each epoch.",
    )
    parser.add_argument(
        "--grad_probe_interval",
        type=int,
        default=0,
        help="If >0, probe rec/emb term gradient norms every N train batches. 0 disables probing.",
    )
    parser.add_argument(
        "--finetune_codec_weights",
        type=str,
        default="",
        help="Optional codec sampling weights in finetune, e.g. 'clean=0.1,aac=1.0,opus=1.6,amrwb=1.8'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split, crop reproducibility, and general training reproducibility.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest checkpoint under output_dir/checkpoints if one exists.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device, e.g. cuda or cpu. If cuda is requested but unavailable, the script falls back to cpu.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Seed Python and Torch RNGs for reproducible experiments."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_log_writer(log_path: Path):
    """Create a simple logger that writes to stdout and a file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return _log


def append_epoch_record(jsonl_path: Path, record: dict):
    """Append one epoch record as JSON line for incremental recovery."""

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_dataloaders(args):
    """Build train/valid data loaders from one or two manifests."""

    if args.train_feature_manifest:
        train_feature_manifest = Path(args.train_feature_manifest).resolve()
        valid_feature_manifest = Path(args.valid_feature_manifest).resolve() if args.valid_feature_manifest else None

        if valid_feature_manifest and valid_feature_manifest.exists():
            train_set = PrecomputedPairFeatureDataset(train_feature_manifest, max_frames=args.max_frames, random_crop=False)
            valid_set = PrecomputedPairFeatureDataset(valid_feature_manifest, max_frames=args.max_frames, random_crop=False)
        else:
            # Build two deterministic dataset views on the same manifest.
            train_full = PrecomputedPairFeatureDataset(train_feature_manifest, max_frames=args.max_frames, random_crop=False)
            valid_full = PrecomputedPairFeatureDataset(train_feature_manifest, max_frames=args.max_frames, random_crop=False)
            valid_len = max(1, int(len(train_full) * args.valid_ratio))
            train_len = max(1, len(train_full) - valid_len)
            generator = torch.Generator().manual_seed(args.seed)
            indices = torch.randperm(len(train_full), generator=generator).tolist()
            train_indices = indices[:train_len]
            valid_indices = indices[train_len:]
            train_set = Subset(train_full, train_indices)
            valid_set = Subset(valid_full, valid_indices)
    else:
        if not args.train_manifest:
            raise ValueError("Either --train_feature_manifest or --train_manifest must be provided.")

        train_manifest = Path(args.train_manifest).resolve()
        valid_manifest = Path(args.valid_manifest).resolve() if args.valid_manifest else None

        if valid_manifest and valid_manifest.exists():
            train_set = PairFeatureDataset(train_manifest, sample_rate=args.sample_rate, max_frames=args.max_frames, random_crop=False)
            valid_set = PairFeatureDataset(valid_manifest, sample_rate=args.sample_rate, max_frames=args.max_frames, random_crop=False)
        else:
            # Same split indices are reused across two deterministic views.
            train_full = PairFeatureDataset(train_manifest, sample_rate=args.sample_rate, max_frames=args.max_frames, random_crop=False)
            valid_full = PairFeatureDataset(train_manifest, sample_rate=args.sample_rate, max_frames=args.max_frames, random_crop=False)
            valid_len = max(1, int(len(train_full) * args.valid_ratio))
            train_len = max(1, len(train_full) - valid_len)
            generator = torch.Generator().manual_seed(args.seed)
            indices = torch.randperm(len(train_full), generator=generator).tolist()
            train_indices = indices[:train_len]
            valid_indices = indices[train_len:]
            train_set = Subset(train_full, train_indices)
            valid_set = Subset(valid_full, valid_indices)

    train_set = maybe_limit_dataset(train_set, args.max_train_samples, args.seed)
    valid_set = maybe_limit_dataset(valid_set, args.max_valid_samples, args.seed + 1)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pair_batch,
    )

    finetune_train_loader = train_loader
    finetune_codec_weights = parse_codec_weight_map(args.finetune_codec_weights)
    if finetune_codec_weights:
        sampler = build_finetune_codec_sampler(train_set, finetune_codec_weights, args.seed)
        finetune_train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_pair_batch,
        )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pair_batch,
    )
    return train_loader, finetune_train_loader, valid_loader


def maybe_limit_dataset(dataset, max_samples: int, seed: int):
    """Optionally shrink a dataset for smoke tests while keeping sampling reproducible."""

    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    subset, _ = random_split(dataset, [max_samples, len(dataset) - max_samples], generator=generator)
    return subset


def parse_codec_weight_map(text: str) -> dict[str, float]:
    """Parse finetune codec weights from a 'k=v,k=v' string."""

    if not text.strip():
        return {}
    codec_weights: dict[str, float] = {}
    for chunk in text.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid codec weight item '{piece}'. Expected format codec=weight.")
        codec, weight_str = piece.split("=", 1)
        codec = codec.strip().lower()
        weight = float(weight_str.strip())
        if weight <= 0:
            raise ValueError(f"Codec weight must be > 0, got {codec}={weight}.")
        codec_weights[codec] = weight
    return codec_weights


def parse_codec_vocab(text: str) -> list[str]:
    """Parse codec vocabulary from a comma-separated string."""

    vocab = []
    seen = set()
    for item in text.split(","):
        name = item.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        vocab.append(name)
    if "unknown" not in seen:
        vocab.append("unknown")
    return vocab


def map_codec_ids(codec_names: list[str], codec_to_id: dict[str, int], device: torch.device) -> torch.Tensor:
    """Map codec string labels to integer ids for codec-conditioned frontend."""

    unknown_id = codec_to_id["unknown"]
    ids = [int(codec_to_id.get(str(name).lower(), unknown_id)) for name in codec_names]
    return torch.tensor(ids, dtype=torch.long, device=device)


def sample_codec_name(dataset, index: int) -> str:
    """Resolve one sample's codec label from base or subset datasets."""

    if isinstance(dataset, Subset):
        return sample_codec_name(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, PrecomputedPairFeatureDataset):
        return str(dataset.rows[index].get("codec", "unknown")).lower()
    if isinstance(dataset, PairFeatureDataset):
        return str(dataset.rows[index].codec).lower()
    return "unknown"


def collect_codec_histogram(dataset) -> dict[str, int]:
    """Count sample number per codec label for a dataset view."""

    hist: dict[str, int] = defaultdict(int)
    for idx in range(len(dataset)):
        hist[sample_codec_name(dataset, idx)] += 1
    return dict(sorted(hist.items(), key=lambda item: item[0]))


def build_finetune_codec_sampler(dataset, codec_weights: dict[str, float], seed: int) -> WeightedRandomSampler:
    """Build weighted sampler so finetune can up-weight hard codecs."""

    weights = []
    for idx in range(len(dataset)):
        codec = sample_codec_name(dataset, idx)
        weights.append(float(codec_weights.get(codec, 1.0)))
    generator = torch.Generator().manual_seed(seed + 1234)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_optimizer(args, frontend):
    """Build the requested optimizer for CA-AFC frontend parameters."""

    if args.optimizer == "sgd":
        return torch.optim.SGD(
            frontend.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    return torch.optim.AdamW(frontend.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer, steps_per_epoch: int, total_epochs: int):
    """Build an optional LR scheduler.

    - cosine: per-step linear warmup then cosine decay to min_lr
    - step: epoch-wise step decay every lr_decay_epochs epochs
    - none: no scheduler
    """

    if args.scheduler == "none":
        return None

    if args.scheduler == "step":
        if args.lr_decay_epochs <= 0:
            return None
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_epochs,
            gamma=args.lr_decay_gamma,
        )

    total_steps = max(1, steps_per_epoch * total_epochs)
    warmup_steps = max(0, args.warmup_steps)
    min_lr_scale = max(0.0, min(1.0, args.min_lr / max(args.lr, 1e-12)))

    def lr_lambda(current_step: int):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (current_step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_losses(batch, frontend, campplus, stage: str, args, device: torch.device, codec_to_id: dict[str, int]):
    """Compute the staged CA-AFC objective on one batch."""

    clean_feat = batch["clean_feat"].to(device)
    codec_feat = batch["codec_feat"].to(device)
    aux_feat = batch["aux_feat"].to(device)
    lengths = batch["lengths"].to(device)
    mask = mask_from_lengths(lengths, clean_feat.shape[1])

    codec_ids = map_codec_ids(batch.get("codec", []), codec_to_id, device)
    output = frontend(codec_feat, aux_feat, codec_ids=codec_ids)
    use_element_reduction = args.rec_loss_reduction == "element"
    rec_loss = weighted_reconstruction_loss(
        output.enhanced,
        clean_feat,
        mask,
        normalize_by_bins=use_element_reduction,
    )
    if use_element_reduction:
        rec_loss_per_bin = rec_loss
    else:
        rec_loss_per_bin = rec_loss / float(max(1, clean_feat.shape[-1]))

    # Per-sample diagnostics are used only for logging/analysis and do not
    # change the optimization objective.
    frame_weight = clean_feat.abs().mean(dim=-1, keepdim=True).detach() + 1.0
    sample_abs = (output.enhanced - clean_feat).abs() * frame_weight * mask
    sample_denom = mask.sum(dim=(1, 2)).clamp_min(1e-6)
    if use_element_reduction:
        sample_denom = sample_denom * float(max(1, clean_feat.shape[-1]))
    rec_loss_per_sample = sample_abs.sum(dim=(1, 2)) / sample_denom

    smooth_loss = smoothness_loss(output.residual, mask)
    emb_loss = torch.zeros((), device=device)
    emb_loss_per_sample = torch.zeros_like(rec_loss_per_sample)

    if stage == "finetune":
        # Mask padded frames before feeding CAM++ to avoid embedding supervision
        # being polluted by zero-padding regions.
        clean_masked = clean_feat * mask
        enhanced_masked = output.enhanced * mask
        with torch.no_grad():
            clean_embed = campplus(clean_masked)
        enhanced_embed = campplus(enhanced_masked)
        emb_loss = cosine_embedding_consistency(enhanced_embed, clean_embed)
        emb_loss_per_sample = 1.0 - torch.nn.functional.cosine_similarity(enhanced_embed, clean_embed, dim=-1)

    rec_term = args.lambda_rec * rec_loss
    smooth_term = args.lambda_smooth * smooth_loss
    emb_term = torch.zeros((), device=device)
    emb_weight_scale = torch.ones((), device=device)
    lambda_emb_eff = torch.zeros((), device=device)
    if stage == "finetune":
        lambda_emb_eff = torch.full_like(emb_loss, args.lambda_emb)
        if args.emb_term_target_ratio > 0:
            desired_emb_term = rec_term.detach() * args.emb_term_target_ratio
            current_emb_term = (lambda_emb_eff * emb_loss).detach().clamp_min(1e-8)
            emb_weight_scale = (desired_emb_term / current_emb_term).clamp(
                min=args.emb_lambda_scale_min,
                max=args.emb_lambda_scale_max,
            )
            lambda_emb_eff = lambda_emb_eff * emb_weight_scale
        emb_term = lambda_emb_eff * emb_loss

    total = rec_term + smooth_term + emb_term
    emb_to_rec_ratio = emb_term / rec_term.clamp_min(1e-12)

    return total, {
        "loss": float(total.detach().cpu()),
        "rec_loss": float(rec_loss.detach().cpu()),
        "rec_loss_per_bin": float(rec_loss_per_bin.detach().cpu()),
        "emb_loss": float(emb_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "rec_term": float(rec_term.detach().cpu()),
        "emb_term": float(emb_term.detach().cpu()),
        "lambda_emb_eff": float(lambda_emb_eff.detach().cpu()),
        "emb_weight_scale": float(emb_weight_scale.detach().cpu()),
        "smooth_term": float(smooth_term.detach().cpu()),
        "emb_to_rec_ratio": float(emb_to_rec_ratio.detach().cpu()),
    }, {
        "rec_term": rec_term,
        "emb_term": emb_term,
    }, {
        "rec_loss_per_sample": rec_loss_per_sample.detach().cpu(),
        "emb_loss_per_sample": emb_loss_per_sample.detach().cpu(),
    }


def grad_l2_norm_from_grads(grads) -> float:
    """Compute global L2 norm from a gradient tensor list."""

    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_detached = grad.detach()
        total += float(torch.sum(grad_detached * grad_detached).item())
    if total <= 0.0:
        return 0.0
    return math.sqrt(total)


def probe_term_grad_norm(term: torch.Tensor, parameters) -> float:
    """Measure gradient norm contribution of one objective term to frontend params."""

    if term is None or not term.requires_grad:
        return 0.0
    grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
    return grad_l2_norm_from_grads(grads)


def grad_l2_norm(parameters) -> float:
    """Compute global L2 norm of gradients for quick gradient-health monitoring."""

    grads = [param.grad for param in parameters if param.grad is not None]
    return grad_l2_norm_from_grads(grads)


def run_epoch(
    loader,
    frontend,
    campplus,
    optimizer,
    scheduler,
    stage: str,
    args,
    device: torch.device,
    train: bool,
    global_step: int,
    log_fn,
    codec_to_id: dict[str, int],
):
    """Run a full train or validation epoch and aggregate loss statistics."""

    frontend.train(mode=train)
    stats = {
        "loss": 0.0,
        "rec_loss": 0.0,
        "rec_loss_per_bin": 0.0,
        "emb_loss": 0.0,
        "smooth_loss": 0.0,
        "rec_term": 0.0,
        "emb_term": 0.0,
        "lambda_emb_eff": 0.0,
        "emb_weight_scale": 0.0,
        "smooth_term": 0.0,
        "emb_to_rec_ratio": 0.0,
        "grad_norm": 0.0,
        "grad_rec_norm": 0.0,
        "grad_emb_norm": 0.0,
        "grad_emb_rec_ratio": 0.0,
    }
    trainable_params = [param for param in frontend.parameters() if param.requires_grad]
    num_steps = 0
    total_steps = len(loader)
    split_name = "train" if train else "valid"
    codec_stats = {}
    start_time = time.perf_counter()
    last_lr = optimizer.param_groups[0]["lr"]
    best_batch = {
        "step": -1,
        "loss": float("inf"),
        "rec_loss": 0.0,
        "emb_loss": 0.0,
        "smooth_loss": 0.0,
    }
    worst_batch = {
        "step": -1,
        "loss": float("-inf"),
        "rec_loss": 0.0,
        "emb_loss": 0.0,
        "smooth_loss": 0.0,
    }

    for step, batch in enumerate(loader, start=1):
        with torch.set_grad_enabled(train):
            total, batch_stats, term_tensors, sample_metrics = compute_losses(
                batch,
                frontend,
                campplus,
                stage,
                args,
                device,
                codec_to_id,
            )
            if train:
                should_probe = args.grad_probe_interval > 0 and (
                    step == 1 or step % args.grad_probe_interval == 0 or step == total_steps
                )
                if should_probe:
                    rec_grad_norm = probe_term_grad_norm(term_tensors["rec_term"], trainable_params)
                    emb_grad_norm = probe_term_grad_norm(term_tensors["emb_term"], trainable_params)
                    batch_stats["grad_rec_norm"] = rec_grad_norm
                    batch_stats["grad_emb_norm"] = emb_grad_norm
                    batch_stats["grad_emb_rec_ratio"] = emb_grad_norm / max(rec_grad_norm, 1e-12)
                else:
                    batch_stats["grad_rec_norm"] = 0.0
                    batch_stats["grad_emb_norm"] = 0.0
                    batch_stats["grad_emb_rec_ratio"] = 0.0

                optimizer.zero_grad()
                total.backward()
                batch_stats["grad_norm"] = grad_l2_norm(trainable_params)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
                optimizer.step()
                if scheduler is not None and args.scheduler == "cosine":
                    scheduler.step()
                global_step += 1
                last_lr = optimizer.param_groups[0]["lr"]
            else:
                batch_stats["grad_norm"] = 0.0
                batch_stats["grad_rec_norm"] = 0.0
                batch_stats["grad_emb_norm"] = 0.0
                batch_stats["grad_emb_rec_ratio"] = 0.0

        for key, value in batch_stats.items():
            stats[key] += value

        batch_codecs = batch.get("codec", [])
        rec_samples = sample_metrics["rec_loss_per_sample"].tolist()
        emb_samples = sample_metrics["emb_loss_per_sample"].tolist()
        if len(batch_codecs) == len(rec_samples):
            for codec, rec_val, emb_val in zip(batch_codecs, rec_samples, emb_samples):
                name = str(codec).lower()
                bucket = codec_stats.setdefault(
                    name,
                    {
                        "count": 0,
                        "rec_loss": 0.0,
                        "emb_loss": 0.0,
                        "emb_rec_ratio": 0.0,
                    },
                )
                bucket["count"] += 1
                bucket["rec_loss"] += float(rec_val)
                bucket["emb_loss"] += float(emb_val)
                bucket["emb_rec_ratio"] += float(emb_val) / max(float(rec_val), 1e-12)
        num_steps += 1

        if batch_stats["loss"] < best_batch["loss"]:
            best_batch = {
                "step": step,
                "loss": batch_stats["loss"],
                "rec_loss": batch_stats["rec_loss"],
                "emb_loss": batch_stats["emb_loss"],
                "smooth_loss": batch_stats["smooth_loss"],
            }
        if batch_stats["loss"] > worst_batch["loss"]:
            worst_batch = {
                "step": step,
                "loss": batch_stats["loss"],
                "rec_loss": batch_stats["rec_loss"],
                "emb_loss": batch_stats["emb_loss"],
                "smooth_loss": batch_stats["smooth_loss"],
            }

        if args.log_interval > 0 and (step == 1 or step % args.log_interval == 0 or step == total_steps):
            elapsed = time.perf_counter() - start_time
            avg_step = elapsed / step
            eta = max(0.0, avg_step * (total_steps - step))
            log_fn(
                "[{split}] stage={stage} batch={step}/{total} "
                "loss={loss:.4f} rec={rec:.4f} rec_bin={rec_bin:.4f} emb={emb:.4f} smooth={smooth:.4f} "
                "rec_term={rec_term:.4f} emb_term={emb_term:.4f} emb/rec={emb_to_rec:.4f} "
                "lambda_emb_eff={lambda_emb_eff:.4f} emb_scale={emb_scale:.4f} "
                "grad={grad_norm:.4f} grad_rec={grad_rec_norm:.4f} grad_emb={grad_emb_norm:.4f} grad_emb/rec={grad_emb_rec_ratio:.4f} "
                "lr={lr:.6f} "
                "elapsed={elapsed:.1f}s eta={eta:.1f}s".format(
                    split=split_name,
                    stage=stage,
                    step=step,
                    total=total_steps,
                    loss=batch_stats["loss"],
                    rec=batch_stats["rec_loss"],
                    rec_bin=batch_stats["rec_loss_per_bin"],
                    emb=batch_stats["emb_loss"],
                    smooth=batch_stats["smooth_loss"],
                    rec_term=batch_stats["rec_term"],
                    emb_term=batch_stats["emb_term"],
                    emb_to_rec=batch_stats["emb_to_rec_ratio"],
                    lambda_emb_eff=batch_stats["lambda_emb_eff"],
                    emb_scale=batch_stats["emb_weight_scale"],
                    grad_norm=batch_stats["grad_norm"],
                    grad_rec_norm=batch_stats["grad_rec_norm"],
                    grad_emb_norm=batch_stats["grad_emb_norm"],
                    grad_emb_rec_ratio=batch_stats["grad_emb_rec_ratio"],
                    lr=last_lr,
                    elapsed=elapsed,
                    eta=eta,
                )
            )

    if num_steps == 0:
        return stats, 0.0, global_step, {"best_batch": best_batch, "worst_batch": worst_batch, "codec_stats": {}}

    codec_summary = {}
    for codec, bucket in sorted(codec_stats.items(), key=lambda item: item[0]):
        count = max(1, int(bucket["count"]))
        codec_summary[codec] = {
            "count": int(bucket["count"]),
            "rec_loss": bucket["rec_loss"] / count,
            "emb_loss": bucket["emb_loss"] / count,
            "emb_rec_ratio": bucket["emb_rec_ratio"] / count,
        }

    epoch_seconds = time.perf_counter() - start_time
    return {key: value / num_steps for key, value in stats.items()}, epoch_seconds, global_step, {
        "best_batch": best_batch,
        "worst_batch": worst_batch,
        "codec_stats": codec_summary,
    }


def save_frontend_checkpoint(path: Path, frontend, optimizer, scheduler, epoch: int, stage: str, global_step: int, args):
    """Save frontend weights and optimizer state for resume/testing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frontend_state": frontend.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "stage": stage,
            "global_step": global_step,
            "args": vars(args),
        },
        str(path),
    )


def maybe_resume(frontend, optimizer, scheduler, output_dir: Path):
    """Resume from the newest checkpoint if one exists."""

    ckpt = latest_frontend_checkpoint(output_dir / "checkpoints")
    if ckpt is None:
        return 0, "pretrain", 0
    state = torch.load(str(ckpt), map_location="cpu")
    frontend.load_state_dict(state["frontend_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    return int(state.get("epoch", 0)), str(state.get("stage", "pretrain")), int(state.get("global_step", 0))


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    epoch_jsonl_path = output_dir / "epoch_records.jsonl"
    if not args.resume:
        log_path.write_text("", encoding="utf-8")
        epoch_jsonl_path.write_text("", encoding="utf-8")
    log_fn = build_log_writer(log_path)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    train_loader, finetune_train_loader, valid_loader = make_dataloaders(args)
    train_codec_hist = collect_codec_histogram(train_loader.dataset)
    valid_codec_hist = collect_codec_histogram(valid_loader.dataset)
    finetune_codec_weights = parse_codec_weight_map(args.finetune_codec_weights)
    codec_vocab = parse_codec_vocab(args.codec_vocab)
    codec_to_id = {name: idx for idx, name in enumerate(codec_vocab)}

    frontend = CAAFCFrontend(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        band_scale=args.band_scale,
        residual_scale=args.residual_scale,
        num_codecs=len(codec_vocab),
        codec_emb_dim=args.codec_emb_dim,
    ).to(device)
    campplus = load_frozen_campplus(Path(args.campplus_model_bin).resolve(), device=device)
    campplus.eval()
    optimizer = build_optimizer(args, frontend)
    scheduler = build_scheduler(args, optimizer, steps_per_epoch=len(train_loader), total_epochs=args.pretrain_epochs + args.finetune_epochs)

    start_epoch = 0
    resumed_stage = "pretrain"
    global_step = 0
    if args.resume:
        start_epoch, resumed_stage, global_step = maybe_resume(frontend, optimizer, scheduler, output_dir)

    history = []
    best_valid = float("inf")
    total_epochs = args.pretrain_epochs + args.finetune_epochs
    log_fn(
        "Training setup: device={device}, train_samples={train_samples}, valid_samples={valid_samples}, "
        "train_batches={train_batches}, valid_batches={valid_batches}, total_epochs={epochs}".format(
            device=device,
            train_samples=len(train_loader.dataset),
            valid_samples=len(valid_loader.dataset),
            train_batches=len(train_loader),
            valid_batches=len(valid_loader),
            epochs=total_epochs,
        )
    )
    log_fn("Train codec histogram: " + json.dumps(train_codec_hist, ensure_ascii=False))
    log_fn("Valid codec histogram: " + json.dumps(valid_codec_hist, ensure_ascii=False))
    log_fn("Codec vocabulary: " + json.dumps(codec_vocab, ensure_ascii=False))
    if finetune_codec_weights:
        log_fn("Finetune weighted sampling enabled: " + json.dumps(finetune_codec_weights, ensure_ascii=False))

    for epoch in range(start_epoch + 1, total_epochs + 1):
        stage = "pretrain" if epoch <= args.pretrain_epochs else "finetune"
        if resumed_stage == "finetune" and epoch <= args.pretrain_epochs:
            continue
        frontend.dropout.p = args.pretrain_dropout if stage == "pretrain" else args.dropout

        log_fn(
            "Epoch {epoch}/{total} started: stage={stage}, train_batches={train_batches}, valid_batches={valid_batches}".format(
                epoch=epoch,
                total=total_epochs,
                stage=stage,
                train_batches=len(finetune_train_loader if stage == "finetune" else train_loader),
                valid_batches=len(valid_loader),
            )
        )
        epoch_start = time.perf_counter()
        active_train_loader = finetune_train_loader if stage == "finetune" else train_loader
        train_stats, train_seconds, global_step, train_batch_extremes = run_epoch(
            active_train_loader,
            frontend,
            campplus,
            optimizer,
            scheduler,
            stage,
            args,
            device,
            train=True,
            global_step=global_step,
            log_fn=log_fn,
            codec_to_id=codec_to_id,
        )
        valid_stats, valid_seconds, global_step, valid_batch_extremes = run_epoch(
            valid_loader,
            frontend,
            campplus,
            optimizer,
            scheduler,
            stage,
            args,
            device,
            train=False,
            global_step=global_step,
            log_fn=log_fn,
            codec_to_id=codec_to_id,
        )
        if scheduler is not None and args.scheduler == "step":
            scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        remaining_epochs = total_epochs - epoch
        eta_seconds = epoch_seconds * remaining_epochs

        record = {
            "epoch": epoch,
            "stage": stage,
            "train": train_stats,
            "valid": valid_stats,
            "batch_extremes": {
                "train": train_batch_extremes,
                "valid": valid_batch_extremes,
            },
            "timing": {
                "train_seconds": train_seconds,
                "valid_seconds": valid_seconds,
                "epoch_seconds": epoch_seconds,
                "remaining_epochs": remaining_epochs,
                "eta_seconds": eta_seconds,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
            },
        }
        history.append(record)
        append_epoch_record(epoch_jsonl_path, record)
        log_fn("epoch_record=" + json.dumps(record, ensure_ascii=False))
        log_fn(
            "Epoch {epoch}/{total} finished in {epoch_seconds:.1f}s "
            "(train={train_seconds:.1f}s, valid={valid_seconds:.1f}s), remaining_eta={eta_seconds:.1f}s ({eta_minutes:.1f} min)".format(
                epoch=epoch,
                total=total_epochs,
                epoch_seconds=epoch_seconds,
                train_seconds=train_seconds,
                valid_seconds=valid_seconds,
                eta_seconds=eta_seconds,
                eta_minutes=eta_seconds / 60.0,
            )
        )

        ckpt_path = output_dir / "checkpoints" / f"ca_afc_epoch_{epoch:03d}.pt"
        save_frontend_checkpoint(ckpt_path, frontend, optimizer, scheduler, epoch, stage, global_step, args)

        if valid_stats["loss"] < best_valid:
            best_valid = valid_stats["loss"]
            save_frontend_checkpoint(output_dir / "best_frontend.pt", frontend, optimizer, scheduler, epoch, stage, global_step, args)

    save_json(
        output_dir / "train_summary.json",
        {
            "args": vars(args),
            "best_valid_loss": best_valid,
            "history": history,
            "epoch_jsonl": str(epoch_jsonl_path),
            "log_file": str(log_path),
        },
    )
    log_fn(f"Saved best frontend checkpoint: {output_dir / 'best_frontend.pt'}")


if __name__ == "__main__":
    main()
