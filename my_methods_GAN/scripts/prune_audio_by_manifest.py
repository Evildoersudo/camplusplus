#!/usr/bin/env python3
"""Delete audio files under a root unless they are listed in a manifest.

The default mode is dry-run. Permanent deletion requires --delete.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_root", type=Path, required=True)
    parser.add_argument("--keep_manifest", type=Path, required=True)
    parser.add_argument(
        "--extensions",
        default=".m4a",
        help="Comma-separated extensions eligible for deletion. Default: .m4a",
    )
    parser.add_argument(
        "--deleted_list_out",
        type=Path,
        default=None,
        help="Text file recording paths selected for deletion.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete unlisted files. Without this flag, only report.",
    )
    parser.add_argument(
        "--remove_empty_dirs",
        action="store_true",
        help="After deletion, remove empty directories below audio_root.",
    )
    return parser.parse_args()


def normalize_extensions(raw: str) -> set[str]:
    extensions = {
        value.strip().lower()
        if value.strip().startswith(".")
        else f".{value.strip().lower()}"
        for value in raw.split(",")
        if value.strip()
    }
    if not extensions:
        raise ValueError("--extensions must contain at least one extension")
    return extensions


def relative_to_root(path_text: str, audio_root: Path) -> str:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(audio_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Manifest path is outside --audio_root: {resolved}"
            ) from exc

    rel = Path(path)
    if ".." in rel.parts:
        raise ValueError(f"Manifest relative path escapes audio root: {path_text}")
    return rel.as_posix()


def load_keep_paths(manifest: Path, audio_root: Path) -> set[str]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "clean_wav" not in fields and "rel_path" not in fields:
            raise ValueError(
                f"Manifest must contain clean_wav or rel_path: {manifest}"
            )
        keep: set[str] = set()
        for row_number, row in enumerate(reader, 2):
            path_text = row.get("clean_wav") or row.get("rel_path")
            if not path_text:
                raise ValueError(f"Missing clean path at manifest row {row_number}")
            keep.add(relative_to_root(path_text, audio_root))
    if not keep:
        raise ValueError(f"Manifest contains no keep paths: {manifest}")
    return keep


def remove_empty_directories(audio_root: Path) -> int:
    removed = 0
    directories = sorted(
        (path for path in audio_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def main() -> int:
    args = parse_args()
    audio_root = args.audio_root.resolve()
    manifest = args.keep_manifest.resolve()
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing audio root: {audio_root}")
    if audio_root == Path("/"):
        raise ValueError("Refusing to operate on filesystem root")
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing keep manifest: {manifest}")

    extensions = normalize_extensions(args.extensions)
    keep_relpaths = load_keep_paths(manifest, audio_root)
    audio_files = sorted(
        path
        for path in audio_root.glob("*/*/*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not audio_files:
        raise RuntimeError(
            f"No id/video/audio files with extensions {sorted(extensions)} under {audio_root}"
        )

    existing_relpaths = {
        path.relative_to(audio_root).as_posix(): path for path in audio_files
    }
    missing_keep = sorted(keep_relpaths - set(existing_relpaths))
    if missing_keep:
        preview = "\n".join(missing_keep[:10])
        raise RuntimeError(
            f"{len(missing_keep)} manifest files are missing below {audio_root}; "
            f"refusing to delete anything. First missing paths:\n{preview}"
        )

    delete_paths = [
        path
        for relpath, path in existing_relpaths.items()
        if relpath not in keep_relpaths
    ]
    delete_paths.sort()
    delete_bytes = sum(path.stat().st_size for path in delete_paths)
    keep_bytes = sum(existing_relpaths[rel].stat().st_size for rel in keep_relpaths)

    deleted_list = (
        args.deleted_list_out.resolve()
        if args.deleted_list_out
        else manifest.with_suffix(manifest.suffix + ".deleted_files.txt")
    )
    deleted_list.parent.mkdir(parents=True, exist_ok=True)
    recorded_paths: set[str] = set()
    if deleted_list.is_file():
        recorded_paths = {
            line.strip()
            for line in deleted_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    recorded_paths.update(str(path) for path in delete_paths)
    with deleted_list.open("w", encoding="utf-8") as handle:
        for path_text in sorted(recorded_paths):
            handle.write(path_text + "\n")

    print(f"Audio root:            {audio_root}")
    print(f"Keep manifest:         {manifest}")
    print(f"Eligible audio files:  {len(audio_files)}")
    print(f"Files to keep:         {len(keep_relpaths)}")
    print(f"Files to delete:       {len(delete_paths)}")
    print(f"Bytes to keep:         {keep_bytes / (1024 ** 3):.2f} GiB")
    print(f"Bytes to delete:       {delete_bytes / (1024 ** 3):.2f} GiB")
    print(f"Deletion list:         {deleted_list}")

    if not args.delete:
        print("Dry run only; no audio files were deleted. Add --delete to apply.")
        return 0

    for index, path in enumerate(delete_paths, 1):
        path.unlink()
        if index % 10000 == 0 or index == len(delete_paths):
            print(f"Deleted:               {index}/{len(delete_paths)}")

    if args.remove_empty_dirs:
        removed = remove_empty_directories(audio_root)
        print(f"Empty directories:     removed {removed}")

    remaining = sum(
        1
        for path in audio_root.glob("*/*/*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if remaining != len(keep_relpaths):
        raise RuntimeError(
            f"Post-delete count mismatch: remaining={remaining}, expected={len(keep_relpaths)}"
        )
    print(f"Done. Remaining files: {remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
