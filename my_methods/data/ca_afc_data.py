"""Feature extraction, dataset, and checkpoint helpers for CA-AFC.

The functions here are intentionally reusable across:
- pair-manifest creation
- frontend training
- frontend + CAM++ evaluation
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from speakerlab.models.campplus.DTDNN import CAMPPlus


def load_wav_mono(path: Path, sample_rate: int) -> torch.Tensor:
    """Load mono waveform and resample to the target sample rate if needed."""

    wav, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if wav.shape[0] > 1:
        wav = wav[:1]
    return wav.squeeze(0)


def compute_fbank(wav: torch.Tensor, sample_rate: int, n_mels: int = 80, mean_norm: bool = True) -> torch.Tensor:
    """Compute Kaldi-compatible log-Mel FBank features."""

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    feat = Kaldi.fbank(wav, num_mel_bins=n_mels, sample_frequency=sample_rate, dither=0.0)
    if mean_norm:
        feat = feat - feat.mean(dim=0, keepdim=True)
    return feat


def compute_aux_features(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Compute auxiliary temporal cues.

    Returns `[pitch, delta_pitch, voiced_flag]` with the same frame count as the
    later FBank after alignment.
    """

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    pitch = torchaudio.functional.detect_pitch_frequency(
        wav,
        sample_rate=sample_rate,
        frame_time=0.01,
        win_length=30,
    ).squeeze(0)
    pitch = torch.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0)
    voiced = (pitch > 1.0).float()
    delta_pitch = torch.zeros_like(pitch)
    if pitch.numel() > 1:
        delta_pitch[1:] = pitch[1:] - pitch[:-1]

    # Normalize active pitch frames only; silent/unvoiced regions stay near zero.
    active = voiced > 0
    if active.any():
        mean = pitch[active].mean()
        std = pitch[active].std().clamp_min(1e-5)
        pitch = torch.where(active, (pitch - mean) / std, torch.zeros_like(pitch))
        delta_pitch = torch.where(active, delta_pitch / std, torch.zeros_like(delta_pitch))

    return torch.stack([pitch, delta_pitch, voiced], dim=-1)


def align_feature_lengths(*features: torch.Tensor) -> List[torch.Tensor]:
    """Trim multiple time-major features to the minimum shared frame count."""

    min_len = min(feat.shape[0] for feat in features)
    return [feat[:min_len] for feat in features]


def crop_or_pad_pair(
    clean_feat: torch.Tensor,
    codec_feat: torch.Tensor,
    aux_feat: torch.Tensor,
    max_frames: int,
    random_crop: bool,
) -> List[torch.Tensor]:
    """Crop or pad the paired features to a uniform training length."""

    length = clean_feat.shape[0]
    if max_frames <= 0:
        return [clean_feat, codec_feat, aux_feat]

    if length > max_frames:
        start = random.randint(0, length - max_frames) if random_crop else 0
        end = start + max_frames
        return [clean_feat[start:end], codec_feat[start:end], aux_feat[start:end]]

    pad_len = max_frames - length
    if pad_len <= 0:
        return [clean_feat, codec_feat, aux_feat]

    clean_pad = torch.nn.functional.pad(clean_feat, (0, 0, 0, pad_len))
    codec_pad = torch.nn.functional.pad(codec_feat, (0, 0, 0, pad_len))
    aux_pad = torch.nn.functional.pad(aux_feat, (0, 0, 0, pad_len))
    return [clean_pad, codec_pad, aux_pad]


def mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create a `[B, T]` mask for valid frames."""

    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def weighted_reconstruction_loss(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    mask: torch.Tensor,
    band_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted MSE loss, optionally emphasizing low/mid frequency bands."""

    if band_weights is None:
        band_weights = torch.ones(clean.shape[-1], device=clean.device, dtype=clean.dtype)
        band_weights[:20] = 1.6
        band_weights[20:50] = 1.3

    diff = (enhanced - clean).pow(2) * band_weights.view(1, 1, -1)
    diff = diff * mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1).to(clean.dtype) * clean.shape[-1]
    return diff.sum() / denom


def smoothness_loss(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Temporal smoothness loss on adjacent frames."""

    if sequence.shape[1] <= 1:
        return torch.zeros((), device=sequence.device, dtype=sequence.dtype)
    diff = (sequence[:, 1:] - sequence[:, :-1]).abs()
    pair_mask = mask[:, 1:] & mask[:, :-1]
    denom = pair_mask.sum().clamp_min(1).to(sequence.dtype) * sequence.shape[-1]
    return (diff * pair_mask.unsqueeze(-1)).sum() / denom


def cosine_embedding_consistency(enhanced_embed: torch.Tensor, clean_embed: torch.Tensor) -> torch.Tensor:
    """Speaker embedding consistency loss used for stage-2 task-oriented tuning."""

    enhanced_embed = torch.nn.functional.normalize(enhanced_embed, dim=-1)
    clean_embed = torch.nn.functional.normalize(clean_embed, dim=-1)
    return 1.0 - torch.nn.functional.cosine_similarity(enhanced_embed, clean_embed, dim=-1).mean()


def load_frozen_campplus(
    model_bin: Path,
    device: torch.device,
    feat_dim: int = 80,
    embedding_size: int = 192,
) -> CAMPPlus:
    """Load pretrained CAM++ and freeze all backend parameters."""

    model = CAMPPlus(feat_dim=feat_dim, embedding_size=embedding_size)
    state = torch.load(str(model_bin), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


@dataclass
class PairSample:
    """Single pair-manifest row after CSV parsing."""

    utt_id: str
    spk_id: str
    clean_wav: Path
    codec_wav: Path
    condition: str
    codec: str
    bitrate: str


def read_pair_manifest(path: Path) -> List[PairSample]:
    """Load the pair manifest produced by `build_pair_manifest.py`."""

    rows: List[PairSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                PairSample(
                    utt_id=row["utt_id"],
                    spk_id=row["spk_id"],
                    clean_wav=Path(row["clean_wav"]).resolve(),
                    codec_wav=Path(row["codec_wav"]).resolve(),
                    condition=row.get("condition", "unknown"),
                    codec=row.get("codec", "unknown"),
                    bitrate=row.get("bitrate", "-"),
                )
            )
    return rows


def extract_pair_features(clean_wav_path: Path, codec_wav_path: Path, sample_rate: int) -> Dict[str, torch.Tensor]:
    """Extract aligned clean/codec/aux features for one pair and return tensors."""

    clean_wav = load_wav_mono(clean_wav_path, sample_rate)
    codec_wav = load_wav_mono(codec_wav_path, sample_rate)

    clean_feat = compute_fbank(clean_wav, sample_rate)
    codec_feat = compute_fbank(codec_wav, sample_rate)
    aux_feat = compute_aux_features(codec_wav, sample_rate)
    clean_feat, codec_feat, aux_feat = align_feature_lengths(clean_feat, codec_feat, aux_feat)
    return {
        "clean_feat": clean_feat.contiguous(),
        "codec_feat": codec_feat.contiguous(),
        "aux_feat": aux_feat.contiguous(),
        "length": torch.tensor(clean_feat.shape[0], dtype=torch.long),
    }


@dataclass
class PrecomputedFeatureSample:
    """Single precomputed-feature manifest row."""

    utt_id: str
    spk_id: str
    feature_path: Path
    length: int
    condition: str
    codec: str
    bitrate: str


def read_precomputed_feature_manifest(path: Path) -> List[PrecomputedFeatureSample]:
    """Load the manifest produced by `precompute_pair_features.py`."""

    rows: List[PrecomputedFeatureSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                PrecomputedFeatureSample(
                    utt_id=row["utt_id"],
                    spk_id=row["spk_id"],
                    feature_path=Path(row["feature_pt"]).resolve(),
                    length=int(row["length"]),
                    condition=row.get("condition", "unknown"),
                    codec=row.get("codec", "unknown"),
                    bitrate=row.get("bitrate", "-"),
                )
            )
    return rows


class PairFeatureDataset(Dataset):
    """Load `(clean_wav, codec_wav)` pairs and extract training features."""

    def __init__(
        self,
        manifest_path: Path,
        sample_rate: int = 16000,
        max_frames: int = 300,
        random_crop: bool = True,
    ):
        self.rows = read_pair_manifest(manifest_path)
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        feature_pack = extract_pair_features(row.clean_wav, row.codec_wav, self.sample_rate)
        clean_feat = feature_pack["clean_feat"]
        codec_feat = feature_pack["codec_feat"]
        aux_feat = feature_pack["aux_feat"]
        original_length = int(feature_pack["length"])
        clean_feat, codec_feat, aux_feat = crop_or_pad_pair(
            clean_feat,
            codec_feat,
            aux_feat,
            max_frames=self.max_frames,
            random_crop=self.random_crop,
        )
        length = min(original_length, self.max_frames) if self.max_frames > 0 else original_length

        return {
            "clean_feat": clean_feat,
            "codec_feat": codec_feat,
            "aux_feat": aux_feat,
            "length": torch.tensor(length, dtype=torch.long),
            "utt_id": row.utt_id,
            "spk_id": row.spk_id,
        }


class PrecomputedPairFeatureDataset(Dataset):
    """Load pre-extracted `clean_feat / codec_feat / aux_feat` tensors from `.pt` files."""

    def __init__(self, manifest_path: Path, max_frames: int = 300, random_crop: bool = True):
        self.rows = read_precomputed_feature_manifest(manifest_path)
        self.max_frames = max_frames
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        payload = torch.load(str(row.feature_path), map_location="cpu")
        clean_feat = payload["clean_feat"].float()
        codec_feat = payload["codec_feat"].float()
        aux_feat = payload["aux_feat"].float()
        original_length = int(payload.get("length", clean_feat.shape[0]))
        clean_feat, codec_feat, aux_feat = crop_or_pad_pair(
            clean_feat,
            codec_feat,
            aux_feat,
            max_frames=self.max_frames,
            random_crop=self.random_crop,
        )
        length = min(original_length, self.max_frames) if self.max_frames > 0 else original_length
        return {
            "clean_feat": clean_feat,
            "codec_feat": codec_feat,
            "aux_feat": aux_feat,
            "length": torch.tensor(length, dtype=torch.long),
            "utt_id": row.utt_id,
            "spk_id": row.spk_id,
        }


def collate_pair_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a variable-length feature batch and keep frame lengths."""

    clean = pad_sequence([item["clean_feat"] for item in batch], batch_first=True)
    codec = pad_sequence([item["codec_feat"] for item in batch], batch_first=True)
    aux = pad_sequence([item["aux_feat"] for item in batch], batch_first=True)
    lengths = torch.stack([item["length"] for item in batch], dim=0)
    return {
        "clean_feat": clean,
        "codec_feat": codec,
        "aux_feat": aux,
        "lengths": lengths,
        "utt_ids": [item["utt_id"] for item in batch],
        "spk_ids": [item["spk_id"] for item in batch],
    }


def save_json(path: Path, data: Dict) -> None:
    """Write JSON with UTF-8 and pretty indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def latest_frontend_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return the newest `.pt` checkpoint under a directory."""

    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None
