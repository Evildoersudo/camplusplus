"""Train the CA-AFC frontend with a frozen CAM++ backend.

Training follows the two-stage plan in `my_methods/note/method_way.md`:
1. Reconstruction pretraining.
2. Task-oriented tuning with frozen CAM++ embedding feedback.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

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
        help="AdamW learning rate for CA-AFC parameters.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay coefficient used to regularize CA-AFC parameters.",
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
        "--lambda_smooth",
        type=float,
        default=0.01,
        help="Weight of the temporal smoothness regularizer applied to the predicted residual.",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio when valid_manifest is not provided. Example: 0.1 means 10%% validation.",
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


def make_dataloaders(args):
    """Build train/valid data loaders from one or two manifests."""

    if args.train_feature_manifest:
        train_feature_manifest = Path(args.train_feature_manifest).resolve()
        valid_feature_manifest = Path(args.valid_feature_manifest).resolve() if args.valid_feature_manifest else None

        if valid_feature_manifest and valid_feature_manifest.exists():
            train_set = PrecomputedPairFeatureDataset(train_feature_manifest, max_frames=args.max_frames, random_crop=True)
            valid_set = PrecomputedPairFeatureDataset(valid_feature_manifest, max_frames=args.max_frames, random_crop=False)
        else:
            full_set = PrecomputedPairFeatureDataset(train_feature_manifest, max_frames=args.max_frames, random_crop=True)
            valid_len = max(1, int(len(full_set) * args.valid_ratio))
            train_len = max(1, len(full_set) - valid_len)
            generator = torch.Generator().manual_seed(args.seed)
            train_set, valid_set = random_split(full_set, [train_len, len(full_set) - train_len], generator=generator)
    else:
        if not args.train_manifest:
            raise ValueError("Either --train_feature_manifest or --train_manifest must be provided.")

        train_manifest = Path(args.train_manifest).resolve()
        valid_manifest = Path(args.valid_manifest).resolve() if args.valid_manifest else None

        if valid_manifest and valid_manifest.exists():
            train_set = PairFeatureDataset(train_manifest, sample_rate=args.sample_rate, max_frames=args.max_frames, random_crop=True)
            valid_set = PairFeatureDataset(valid_manifest, sample_rate=args.sample_rate, max_frames=args.max_frames, random_crop=False)
        else:
            full_set = PairFeatureDataset(train_manifest, sample_rate=args.sample_rate, max_frames=args.max_frames, random_crop=True)
            valid_len = max(1, int(len(full_set) * args.valid_ratio))
            train_len = max(1, len(full_set) - valid_len)
            generator = torch.Generator().manual_seed(args.seed)
            train_set, valid_set = random_split(full_set, [train_len, len(full_set) - train_len], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
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
    return train_loader, valid_loader


def compute_losses(batch, frontend, campplus, stage: str, args, device: torch.device):
    """Compute the staged CA-AFC objective on one batch."""

    clean_feat = batch["clean_feat"].to(device)
    codec_feat = batch["codec_feat"].to(device)
    aux_feat = batch["aux_feat"].to(device)
    lengths = batch["lengths"].to(device)
    mask = mask_from_lengths(lengths, clean_feat.shape[1])

    output = frontend(codec_feat, aux_feat)
    rec_loss = weighted_reconstruction_loss(output.enhanced, clean_feat, mask)
    smooth_loss = smoothness_loss(output.residual, mask)
    emb_loss = torch.zeros((), device=device)

    if stage == "finetune":
        clean_embed = campplus(clean_feat)
        enhanced_embed = campplus(output.enhanced)
        emb_loss = cosine_embedding_consistency(enhanced_embed, clean_embed)

    total = args.lambda_rec * rec_loss + args.lambda_smooth * smooth_loss
    if stage == "finetune":
        total = total + args.lambda_emb * emb_loss

    return total, {
        "loss": float(total.detach().cpu()),
        "rec_loss": float(rec_loss.detach().cpu()),
        "emb_loss": float(emb_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
    }


def run_epoch(loader, frontend, campplus, optimizer, stage: str, args, device: torch.device, train: bool):
    """Run a full train or validation epoch and aggregate loss statistics."""

    frontend.train(mode=train)
    stats = {"loss": 0.0, "rec_loss": 0.0, "emb_loss": 0.0, "smooth_loss": 0.0}
    num_steps = 0

    for batch in loader:
        with torch.set_grad_enabled(train):
            total, batch_stats = compute_losses(batch, frontend, campplus, stage, args, device)
            if train:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

        for key, value in batch_stats.items():
            stats[key] += value
        num_steps += 1

    if num_steps == 0:
        return stats
    return {key: value / num_steps for key, value in stats.items()}


def save_frontend_checkpoint(path: Path, frontend, optimizer, epoch: int, stage: str, args):
    """Save frontend weights and optimizer state for resume/testing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frontend_state": frontend.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "args": vars(args),
        },
        str(path),
    )


def maybe_resume(frontend, optimizer, output_dir: Path):
    """Resume from the newest checkpoint if one exists."""

    ckpt = latest_frontend_checkpoint(output_dir / "checkpoints")
    if ckpt is None:
        return 0, "pretrain"
    state = torch.load(str(ckpt), map_location="cpu")
    frontend.load_state_dict(state["frontend_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return int(state.get("epoch", 0)), str(state.get("stage", "pretrain"))


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    train_loader, valid_loader = make_dataloaders(args)
    frontend = CAAFCFrontend(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    campplus = load_frozen_campplus(Path(args.campplus_model_bin).resolve(), device=device)
    optimizer = torch.optim.AdamW(frontend.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    resumed_stage = "pretrain"
    if args.resume:
        start_epoch, resumed_stage = maybe_resume(frontend, optimizer, output_dir)

    history = []
    best_valid = float("inf")
    total_epochs = args.pretrain_epochs + args.finetune_epochs

    for epoch in range(start_epoch + 1, total_epochs + 1):
        stage = "pretrain" if epoch <= args.pretrain_epochs else "finetune"
        if resumed_stage == "finetune" and epoch <= args.pretrain_epochs:
            continue

        train_stats = run_epoch(train_loader, frontend, campplus, optimizer, stage, args, device, train=True)
        valid_stats = run_epoch(valid_loader, frontend, campplus, optimizer, stage, args, device, train=False)

        record = {
            "epoch": epoch,
            "stage": stage,
            "train": train_stats,
            "valid": valid_stats,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        ckpt_path = output_dir / "checkpoints" / f"ca_afc_epoch_{epoch:03d}.pt"
        save_frontend_checkpoint(ckpt_path, frontend, optimizer, epoch, stage, args)

        if valid_stats["loss"] < best_valid:
            best_valid = valid_stats["loss"]
            save_frontend_checkpoint(output_dir / "best_frontend.pt", frontend, optimizer, epoch, stage, args)

    save_json(
        output_dir / "train_summary.json",
        {
            "args": vars(args),
            "best_valid_loss": best_valid,
            "history": history,
        },
    )
    print(f"Saved best frontend checkpoint: {output_dir / 'best_frontend.pt'}")


if __name__ == "__main__":
    main()
