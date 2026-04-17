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
        self.rows = rows
        self.sample_rate = int(sample_rate)
        self.segment_len = int(sample_rate * segment_seconds)
        self.random_crop = bool(random_crop)
        self.expand_multi_codec = bool(expand_multi_codec)
        self.codec_shift_samples = {str(k).lower(): int(v) for k, v in (codec_shift_samples or {}).items()}
        self._rng = random.Random(sample_seed)

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
    }
