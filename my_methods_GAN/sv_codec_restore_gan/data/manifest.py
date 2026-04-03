from __future__ import annotations

import csv
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def _iter_audio_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXT:
            yield path


def _build_rel_index(root: Path) -> dict[str, Path]:
    idx = {}
    for path in _iter_audio_files(root):
        rel = path.relative_to(root).as_posix()
        idx[rel] = path
    return idx


def _infer_spk_id(rel_path: str) -> str:
    parts = rel_path.split("/")
    return parts[0] if parts else "unknown"


def build_pair_manifest(clean_root: str | Path, coded_root: str | Path, output_csv: str | Path) -> int:
    clean_root = Path(clean_root).resolve()
    coded_root = Path(coded_root).resolve()
    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    clean_idx = _build_rel_index(clean_root)
    coded_idx = _build_rel_index(coded_root)
    rel_keys = sorted(set(clean_idx.keys()) & set(coded_idx.keys()))

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "spk_id", "clean_wav", "codec_wav"])
        for rel in rel_keys:
            utt_id = Path(rel).with_suffix("").as_posix().replace("/", "-")
            spk_id = _infer_spk_id(rel)
            writer.writerow([utt_id, spk_id, str(clean_idx[rel]), str(coded_idx[rel])])

    return len(rel_keys)
