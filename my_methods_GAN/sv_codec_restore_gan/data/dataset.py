from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from sv_codec_restore_gan.utils.audio import crop_or_pad, load_audio_mono


@dataclass(frozen=True)
class PairRow:
    utt_id: str
    spk_id: str
    clean_wav: str
    codec_wav: str


def _read_manifest(path: str | Path) -> list[PairRow]:
    rows: list[PairRow] = []
    with Path(path).resolve().open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                PairRow(
                    utt_id=row["utt_id"],
                    spk_id=row["spk_id"],
                    clean_wav=row["clean_wav"],
                    codec_wav=row["codec_wav"],
                )
            )
    return rows


class SVCodecPairDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        sample_rate: int = 16000,
        segment_seconds: float = 2.0,
        random_crop: bool = True,
    ):
        self.rows = _read_manifest(manifest_csv)
        self.sample_rate = int(sample_rate)
        self.segment_len = int(sample_rate * segment_seconds)
        self.random_crop = bool(random_crop)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        clean = load_audio_mono(row.clean_wav, sample_rate=self.sample_rate)
        coded = load_audio_mono(row.codec_wav, sample_rate=self.sample_rate)

        if self.segment_len > 0:
            if self.random_crop:
                clean = crop_or_pad(clean, self.segment_len)
                coded = crop_or_pad(coded, self.segment_len)
            else:
                clean = clean[: self.segment_len] if clean.numel() >= self.segment_len else crop_or_pad(clean, self.segment_len)
                coded = coded[: self.segment_len] if coded.numel() >= self.segment_len else crop_or_pad(coded, self.segment_len)

        return {
            "utt_id": row.utt_id,
            "spk_id": row.spk_id,
            "clean": clean,
            "coded": coded,
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
    }
