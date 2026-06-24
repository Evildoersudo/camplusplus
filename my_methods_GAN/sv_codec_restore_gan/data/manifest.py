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


def _infer_codec_type(path: Path) -> str:
    name = path.name.lower()
    if "amrwb" in name or "amr_wb" in name:
        return "amrwb"
    if "g711" in name and "mulaw" in name:
        return "g711_mulaw"
    if "g711" in name and "alaw" in name:
        return "g711_alaw"
    if "mulaw" in name:
        return "g711_mulaw"
    if "alaw" in name:
        return "g711_alaw"
    if "opus" in name:
        return "opus"
    if "aac" in name:
        return "aac"
    return name.replace(" ", "_")


def _parse_coded_roots(coded_roots: str | Path | list[str] | tuple[str, ...]) -> list[Path]:
    if isinstance(coded_roots, (list, tuple)):
        raw_items = [str(x) for x in coded_roots]
    else:
        raw_items = str(coded_roots).replace(";", ",").split(",")
    roots = [Path(x.strip()).resolve() for x in raw_items if x.strip()]
    if not roots:
        raise ValueError("coded_root is empty.")
    return roots


def _load_include_relpaths(include_manifest: str | Path | None, clean_root: Path) -> set[str] | None:
    if include_manifest is None:
        return None
    include_path = Path(include_manifest).resolve()
    relpaths: set[str] = set()
    with include_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {include_path}")
        for row_number, row in enumerate(reader, 2):
            path_text = row.get("rel_path") or row.get("clean_wav")
            if not path_text:
                raise ValueError(
                    f"Manifest row {row_number} must contain rel_path or clean_wav"
                )
            path = Path(path_text)
            if path.is_absolute():
                try:
                    relpaths.add(path.relative_to(clean_root).as_posix())
                except ValueError:
                    if len(path.parts) < 3:
                        raise ValueError(
                            f"Cannot infer id/video/file relative path from {path_text!r}"
                        )
                    relpaths.add(Path(*path.parts[-3:]).as_posix())
            else:
                relpaths.add(path.as_posix())
    if not relpaths:
        raise ValueError(f"No include rows found in {include_path}")
    return relpaths


def build_pair_manifest(
    clean_root: str | Path,
    coded_root: str | Path | list[str] | tuple[str, ...],
    output_csv: str | Path,
    include_manifest: str | Path | None = None,
) -> int:
    clean_root = Path(clean_root).resolve()
    coded_roots = _parse_coded_roots(coded_root)
    output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    clean_idx = _build_rel_index(clean_root)
    coded_indexes = [_build_rel_index(root) for root in coded_roots]
    include_relpaths = _load_include_relpaths(include_manifest, clean_root)

    rel_keys = sorted(clean_idx.keys())
    if include_relpaths is not None:
        rel_keys = [rel for rel in rel_keys if rel in include_relpaths]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "spk_id", "clean_wav", "codec_wav", "codec_type"])
        for rel in rel_keys:
            codec_pairs = []
            for root, idx in zip(coded_roots, coded_indexes):
                if rel in idx:
                    codec_pairs.append((idx[rel], _infer_codec_type(root)))

            if not codec_pairs:
                continue

            utt_id_base = Path(rel).with_suffix("").as_posix().replace("/", "-")
            spk_id = _infer_spk_id(rel)

            for codec_path, codec_type in codec_pairs:
                utt_id = f"{utt_id_base}_{codec_type}"
                writer.writerow([utt_id, spk_id, str(clean_idx[rel]), str(codec_path), codec_type])

    return sum(sum(1 for idx in coded_indexes if rel in idx) for rel in rel_keys)
