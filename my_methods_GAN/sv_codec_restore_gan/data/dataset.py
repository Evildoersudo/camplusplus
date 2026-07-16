from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from sv_codec_restore_gan.utils.audio import load_audio_mono


@dataclass(frozen=True)
class PairRow:
    utt_id: str
    spk_id: str
    clean_wav: str
    codec_wavs: tuple[str, ...]
    codec_type: str
    sample_type: str = "codec"
    deg_target: float = 1.0


def _infer_codec_type_from_path(codec_wav: str) -> str:
    s = str(codec_wav).lower()
    if "amrwb" in s or "amr_wb" in s:
        return "amrwb"
    if "g711" in s and "mulaw" in s:
        return "g711_mulaw"
    if "g711" in s and "alaw" in s:
        return "g711_alaw"
    if "mulaw" in s:
        return "g711_mulaw"
    if "alaw" in s:
        return "g711_alaw"
    if "opus" in s:
        return "opus"
    if "aac" in s:
        return "aac"
    return "unknown"


def _parse_codec_wavs(codec_cell: str) -> tuple[str, ...]:
    parts = [p.strip() for p in str(codec_cell).replace(";", "|").split("|") if p.strip()]
    if not parts:
        raise ValueError("Empty codec_wav field in manifest row.")
    return tuple(parts)


def _read_manifest(path: str | Path, expand_multi_codec: bool = True) -> list[PairRow]:
    rows: list[PairRow] = []
    with Path(path).resolve().open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            codec_wavs = _parse_codec_wavs(row["codec_wav"])
            codec_type = (row.get("codec_type") or "").strip()
            sample_type = (row.get("sample_type") or "codec").strip().lower()
            deg_target_text = (row.get("deg_target") or "").strip()
            deg_target = float(deg_target_text) if deg_target_text else (0.0 if sample_type == "clean" else 1.0)
            if expand_multi_codec and len(codec_wavs) > 1:
                for codec_wav in codec_wavs:
                    one_codec_type = codec_type or _infer_codec_type_from_path(codec_wav)
                    rows.append(
                        PairRow(
                            utt_id=f"{row['utt_id']}_{one_codec_type}",
                            spk_id=row["spk_id"],
                            clean_wav=row["clean_wav"],
                            codec_wavs=(codec_wav,),
                            codec_type=one_codec_type,
                            sample_type=sample_type,
                            deg_target=deg_target,
                        )
                    )
            else:
                one_codec_type = codec_type
                if not one_codec_type and len(codec_wavs) == 1:
                    one_codec_type = _infer_codec_type_from_path(codec_wavs[0])
                rows.append(
                    PairRow(
                        utt_id=row["utt_id"],
                        spk_id=row["spk_id"],
                        clean_wav=row["clean_wav"],
                        codec_wavs=codec_wavs,
                        codec_type=one_codec_type,
                        sample_type=sample_type,
                        deg_target=deg_target,
                    )
                )
    return rows


def _sample_rows_by_speaker(rows: list[PairRow], fraction: float, seed: int) -> list[PairRow]:
    if fraction >= 1.0:
        return rows
    if fraction <= 0.0:
        raise ValueError("sample_fraction must be in (0, 1].")

    rng = random.Random(seed)
    buckets: dict[str, list[PairRow]] = defaultdict(list)
    for row in rows:
        buckets[row.spk_id].append(row)

    sampled: list[PairRow] = []
    for spk, items in buckets.items():
        local = items[:]
        rng.shuffle(local)
        keep = max(1, int(len(local) * fraction))
        sampled.extend(local[:keep])

    rng.shuffle(sampled)
    return sampled


def _add_clean_passthrough_rows(rows: list[PairRow], ratio: float, seed: int) -> list[PairRow]:
    """Append synthetic clean->clean rows.

    ``ratio`` is the desired clean fraction in the final dataset.  For example,
    ratio=0.2 means clean rows should be about 20% of (codec + clean) rows.
    """
    ratio = float(ratio)
    if ratio <= 0.0:
        return rows
    if ratio >= 1.0:
        raise ValueError("clean_passthrough_ratio must be in [0, 1).")
    if not rows:
        return rows

    clean_count = max(1, round(len(rows) * ratio / (1.0 - ratio)))
    rng = random.Random(seed)
    clean_rows: list[PairRow] = []
    for clean_idx in range(clean_count):
        src = rng.choice(rows)
        clean_rows.append(
            PairRow(
                utt_id=f"{src.utt_id}_cleanpt_{clean_idx:08d}",
                spk_id=src.spk_id,
                clean_wav=src.clean_wav,
                codec_wavs=(src.clean_wav,),
                codec_type="clean",
                sample_type="clean",
                deg_target=0.0,
            )
        )

    mixed = rows[:] + clean_rows
    rng.shuffle(mixed)
    return mixed


class SVCodecPairDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        random_crop: bool = True,
        sample_fraction: float = 1.0,
        sample_seed: int = 42,
        stratified_sample: bool = True,
        expand_multi_codec: bool = True,
        codec_shift_samples: dict[str, int] | None = None,
        clean_passthrough_ratio: float = 0.0,
        clean_passthrough_seed: int | None = None,
    ):
        rows = _read_manifest(manifest_csv, expand_multi_codec=expand_multi_codec)
        if sample_fraction < 1.0:
            if stratified_sample:
                rows = _sample_rows_by_speaker(rows, sample_fraction, sample_seed)
            else:
                rng = random.Random(sample_seed)
                shuffled = rows[:]
                rng.shuffle(shuffled)
                keep = max(1, int(len(shuffled) * sample_fraction))
                rows = shuffled[:keep]
        rows = _add_clean_passthrough_rows(
            rows,
            ratio=clean_passthrough_ratio,
            seed=sample_seed if clean_passthrough_seed is None else clean_passthrough_seed,
        )
        self.rows = rows
        self.sample_rate = int(sample_rate)
        self.segment_len = int(sample_rate * segment_seconds)
        self.random_crop = bool(random_crop)
        self.expand_multi_codec = bool(expand_multi_codec)
        self.codec_shift_samples = {str(k).lower(): int(v) for k, v in (codec_shift_samples or {}).items()}
        self._rng = random.Random(sample_seed)

    def sample_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            key = str(row.sample_type or "codec")
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _apply_codec_shift(clean: torch.Tensor, coded: torch.Tensor, shift_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        if shift_samples == 0:
            return clean, coded

        if shift_samples > 0:
            # Positive shift means coded has extra leading delay; drop coded head.
            if coded.numel() <= shift_samples:
                return clean, coded
            coded = coded[shift_samples:]
            clean = clean[: coded.numel()]
            return clean, coded

        # Negative shift means clean has extra leading delay; drop clean head.
        offset = -shift_samples
        if clean.numel() <= offset:
            return clean, coded
        clean = clean[offset:]
        coded = coded[: clean.numel()]
        return clean, coded

    def _aligned_crop_or_pad(self, clean: torch.Tensor, coded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.segment_len <= 0:
            l = min(clean.numel(), coded.numel())
            return clean[:l].contiguous(), coded[:l].contiguous()

        l = min(clean.numel(), coded.numel())
        clean = clean[:l]
        coded = coded[:l]

        if l >= self.segment_len:
            if self.random_crop:
                start = self._rng.randint(0, l - self.segment_len)
            else:
                start = 0
            end = start + self.segment_len
            return clean[start:end].contiguous(), coded[start:end].contiguous()

        pad = self.segment_len - l
        return F.pad(clean, (0, pad)).contiguous(), F.pad(coded, (0, pad)).contiguous()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        clean = load_audio_mono(row.clean_wav, sample_rate=self.sample_rate)
        if len(row.codec_wavs) == 1:
            codec_wav = row.codec_wavs[0]
        elif self.random_crop:
            codec_wav = self._rng.choice(row.codec_wavs)
        else:
            codec_wav = row.codec_wavs[idx % len(row.codec_wavs)]
        is_clean = str(row.sample_type).lower() == "clean" or str(row.codec_type).lower() == "clean"
        if is_clean and str(codec_wav) == str(row.clean_wav):
            coded = clean.clone()
        else:
            coded = load_audio_mono(codec_wav, sample_rate=self.sample_rate)

        shift = self.codec_shift_samples.get(str(row.codec_type).lower(), 0)
        clean, coded = self._apply_codec_shift(clean, coded, shift)

        clean, coded = self._aligned_crop_or_pad(clean, coded)

        return {
            "utt_id": row.utt_id,
            "spk_id": row.spk_id,
            "clean": clean,
            "coded": coded,
            "codec_wav": codec_wav,
            "codec_type": row.codec_type,
            "sample_type": row.sample_type,
            "is_clean": is_clean,
            "deg_target": float(row.deg_target),
            "length": min(clean.numel(), coded.numel()),
        }


def collate_pair_batch(batch: list[dict]) -> dict:
    clean = pad_sequence([item["clean"] for item in batch], batch_first=True)
    coded = pad_sequence([item["coded"] for item in batch], batch_first=True)
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    return {
        "clean": clean,
        "coded": coded,
        "lengths": lengths,
        "utt_id": [item["utt_id"] for item in batch],
        "spk_id": [item["spk_id"] for item in batch],
        "codec_wav": [item["codec_wav"] for item in batch],
        "codec_type": [item.get("codec_type", "") for item in batch],
        "sample_type": [item.get("sample_type", "codec") for item in batch],
        "is_clean": torch.tensor([bool(item.get("is_clean", False)) for item in batch], dtype=torch.bool),
        "deg_target": torch.tensor([float(item.get("deg_target", 1.0)) for item in batch], dtype=torch.float32),
    }
