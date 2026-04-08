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


def _parse_coded_roots(coded_roots: str | Path | list[str] | tuple[str, ...]) -> list[Path]:
    if isinstance(coded_roots, (list, tuple)):
        raw_items = [str(x) for x in coded_roots]
    else:
        raw_items = str(coded_roots).replace(";", ",").split(",")
    roots = [Path(x.strip()).resolve() for x in raw_items if x.strip()]
    if not roots:
        raise ValueError("coded_root is empty.")
    return roots


def build_pair_manifest(clean_root: str | Path, coded_root: str | Path | list[str] | tuple[str, ...], output_csv: str | Path) -> int:
    clean_root = Path(clean_root).resolve()
    coded_roots = _parse_coded_roots(coded_root)
    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    clean_idx = _build_rel_index(clean_root)
    coded_indexes = [_build_rel_index(root) for root in coded_roots]

    rel_keys = sorted(clean_idx.keys())

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "spk_id", "clean_wav", "codec_wav"])
        for rel in rel_keys:
            codec_paths = [str(idx[rel]) for idx in coded_indexes if rel in idx]
            if not codec_paths:
                continue
            utt_id = Path(rel).with_suffix("").as_posix().replace("/", "-")
            spk_id = _infer_spk_id(rel)
            writer.writerow([utt_id, spk_id, str(clean_idx[rel]), "|".join(codec_paths)])

    return sum(1 for rel in rel_keys if any(rel in idx for idx in coded_indexes))
