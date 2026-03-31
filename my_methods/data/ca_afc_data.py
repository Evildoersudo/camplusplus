"""Data helpers for CA-AFC training and evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from my_methods.data.frontend_features import compute_fbank_feature, compute_pitch_proxy, load_audio_mono
from speakerlab.models.campplus.DTDNN import CAMPPlus


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PairManifestRow:
    utt_id: str
    spk_id: str
    clean_wav: str
    codec_wav: str
    condition: str
    codec: str
    bitrate: str


def read_pair_manifest(path: str | Path) -> list[PairManifestRow]:
    rows: list[PairManifestRow] = []
    with Path(path).resolve().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                PairManifestRow(
                    utt_id=row["utt_id"],
                    spk_id=row["spk_id"],
                    clean_wav=str(_resolve_manifest_audio_path(row["clean_wav"])),
                    codec_wav=str(_resolve_manifest_audio_path(row["codec_wav"])),
                    condition=row.get("condition", "unknown"),
                    codec=row.get("codec", "unknown"),
                    bitrate=row.get("bitrate", "-"),
                )
            )
    return rows


def _resolve_manifest_audio_path(path_str: str) -> Path:
    """Resolve manifest audio paths across host/container layouts.

    Pair manifests may be generated on the host and consumed inside Docker.
    In that case, absolute paths such as `/home/dgx/.../camplusplus/...` do not
    exist inside the container even though the path suffix under the repository
    root is still valid. This helper preserves existing paths when possible and
    remaps stale absolute prefixes back under the current repository root.
    """

    path = Path(path_str)
    if path.exists():
        return path.resolve()

    posix_parts = PurePosixPath(path_str).parts
    if "camplusplus" in posix_parts:
        anchor = posix_parts.index("camplusplus")
        suffix_parts = posix_parts[anchor + 1 :]
        remapped = REPO_ROOT.joinpath(*suffix_parts)
        if remapped.exists():
            return remapped.resolve()

    return path.resolve()


def load_wav_mono(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    resolved = _resolve_manifest_audio_path(str(path))
    return load_audio_mono(resolved, sample_rate=sample_rate)


def compute_fbank(wav: torch.Tensor, sample_rate: int = 16000, num_mel_bins: int = 80) -> torch.Tensor:
    return compute_fbank_feature(wav, sample_rate=sample_rate, num_mel_bins=num_mel_bins)


def _frame_energy(wav: torch.Tensor, frame_count: int) -> torch.Tensor:
    if frame_count <= 0:
        return wav.new_zeros((0,))
    frame_len = max(1, wav.numel() // frame_count)
    total = frame_len * frame_count
    if total > wav.numel():
        wav = F.pad(wav, (0, total - wav.numel()))
    else:
        wav = wav[:total]
    frames = wav.view(frame_count, frame_len)
    return frames.pow(2).mean(dim=1).sqrt()


def compute_aux_features(wav: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Compute [pitch, delta_pitch, voiced_flag] aligned to FBank frames."""

    frame_count = compute_fbank(wav, sample_rate=sample_rate).shape[0]
    pitch = compute_pitch_proxy(wav, sample_rate=sample_rate, frame_count=frame_count)
    delta = torch.diff(pitch, dim=0, prepend=pitch[:1])
    energy = _frame_energy(wav, frame_count)
    voiced = ((pitch > 1.0) | (energy > energy.mean())).to(dtype=wav.dtype)
    return torch.stack([pitch, delta, voiced], dim=-1).contiguous()


def _align_pair(clean_feat: torch.Tensor, codec_feat: torch.Tensor, aux_feat: torch.Tensor):
    length = min(clean_feat.shape[0], codec_feat.shape[0], aux_feat.shape[0])
    return clean_feat[:length], codec_feat[:length], aux_feat[:length], int(length)


def extract_pair_features(clean_wav: str | Path, codec_wav: str | Path, sample_rate: int = 16000) -> dict[str, torch.Tensor | int]:
    clean_wav_tensor = load_wav_mono(clean_wav, sample_rate=sample_rate)
    codec_wav_tensor = load_wav_mono(codec_wav, sample_rate=sample_rate)
    clean_feat = compute_fbank(clean_wav_tensor, sample_rate=sample_rate)
    codec_feat = compute_fbank(codec_wav_tensor, sample_rate=sample_rate)
    aux_feat = compute_aux_features(codec_wav_tensor, sample_rate=sample_rate)
    clean_feat, codec_feat, aux_feat, length = _align_pair(clean_feat, codec_feat, aux_feat)
    return {
        "clean_feat": clean_feat,
        "codec_feat": codec_feat,
        "aux_feat": aux_feat,
        "length": length,
    }


def _crop_features(payload: dict[str, torch.Tensor | int], max_frames: int, random_crop: bool):
    clean_feat = payload["clean_feat"]
    codec_feat = payload["codec_feat"]
    aux_feat = payload["aux_feat"]
    length = int(payload["length"])
    if max_frames <= 0 or length <= max_frames:
        return {
            "clean_feat": clean_feat,
            "codec_feat": codec_feat,
            "aux_feat": aux_feat,
            "length": length,
        }

    if random_crop:
        start = int(torch.randint(0, length - max_frames + 1, (1,)).item())
    else:
        start = 0
    end = start + max_frames
    return {
        "clean_feat": clean_feat[start:end],
        "codec_feat": codec_feat[start:end],
        "aux_feat": aux_feat[start:end],
        "length": max_frames,
    }


class PairFeatureDataset(Dataset):
    def __init__(self, manifest_path: str | Path, sample_rate: int = 16000, max_frames: int = 300, random_crop: bool = True):
        self.rows = read_pair_manifest(manifest_path)
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        payload = extract_pair_features(row.clean_wav, row.codec_wav, sample_rate=self.sample_rate)
        payload = _crop_features(payload, self.max_frames, self.random_crop)
        return payload


class PrecomputedPairFeatureDataset(Dataset):
    def __init__(self, manifest_path: str | Path, max_frames: int = 300, random_crop: bool = True):
        self.rows = []
        with Path(manifest_path).resolve().open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.rows.extend(reader)
        self.max_frames = max_frames
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        payload = torch.load(str(Path(self.rows[index]["feature_pt"]).resolve()), map_location="cpu")
        return _crop_features(payload, self.max_frames, self.random_crop)


def collate_pair_batch(batch):
    clean_feat = pad_sequence([item["clean_feat"] for item in batch], batch_first=True)
    codec_feat = pad_sequence([item["codec_feat"] for item in batch], batch_first=True)
    aux_feat = pad_sequence([item["aux_feat"] for item in batch], batch_first=True)
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    return {
        "clean_feat": clean_feat,
        "codec_feat": codec_feat,
        "aux_feat": aux_feat,
        "lengths": lengths,
    }


def mask_from_lengths(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = int(lengths.max().item()) if max_len is None else int(max_len)
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (steps < lengths.unsqueeze(1)).to(dtype=torch.float32).unsqueeze(-1)


def weighted_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    normalize_by_bins: bool = False,
) -> torch.Tensor:
    frame_weight = target.abs().mean(dim=-1, keepdim=True).detach() + 1.0
    loss = (pred - target).abs() * frame_weight * mask
    denom = mask.sum().clamp_min(eps)
    if normalize_by_bins:
        denom = denom * float(max(1, pred.shape[-1]))
    return loss.sum() / denom


def smoothness_loss(residual: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if residual.shape[1] <= 1:
        return residual.new_zeros(())
    diff = residual[:, 1:] - residual[:, :-1]
    diff_mask = mask[:, 1:] * mask[:, :-1]
    return (diff.abs() * diff_mask).sum() / diff_mask.sum().clamp_min(eps)


def cosine_embedding_consistency(enhanced_embed: torch.Tensor, clean_embed: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(enhanced_embed, clean_embed, dim=-1).mean()


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "embedding_model", "frontend_state"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _infer_campplus_embedding_size(state_dict: dict) -> int:
    weight = state_dict.get("xvector.dense.linear.weight")
    if weight is None:
        return 192
    return int(weight.shape[0])


def load_frozen_campplus(model_path: str | Path, device: torch.device | str = "cpu") -> CAMPPlus:
    state = torch.load(str(Path(model_path).resolve()), map_location="cpu")
    state_dict = _unwrap_state_dict(state)
    model = CAMPPlus(feat_dim=80, embedding_size=_infer_campplus_embedding_size(state_dict))
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def latest_frontend_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def save_json(path: str | Path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
