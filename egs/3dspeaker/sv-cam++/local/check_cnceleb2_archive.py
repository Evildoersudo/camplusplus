import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path


def parse_args():
    # This script validates the split CN-Celeb2 download before extraction.
    # It never extracts files to disk. Instead, it checks:
    # 1. whether all split parts exist
    # 2. whether the merged tar.gz can be rebuilt
    # 3. whether the gzip stream and tar headers are readable
    parser = argparse.ArgumentParser(
        description="Check CN-Celeb2 split archives and the merged tar.gz without extracting."
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        required=True,
        help="Directory containing cn-celeb2_v2.tar.gzaa / .gzab / .gzac",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Always rebuild cn-celeb2_v2.tar.gz from split parts before checking",
    )
    parser.add_argument(
        "--chunk_size_mb",
        type=int,
        default=16,
        help="Read chunk size in MB when validating the gzip stream",
    )
    return parser.parse_args()


def md5_of_file(path: Path, chunk_size: int):
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def list_split_parts(download_dir: Path):
    parts = sorted(download_dir.glob("cn-celeb2_v2.tar.gz*"))
    # Keep only true split parts. The merged file itself is handled separately.
    return [part for part in parts if part.name != "cn-celeb2_v2.tar.gz"]


def rebuild_merged_archive(download_dir: Path, parts):
    merged = download_dir / "cn-celeb2_v2.tar.gz"
    merged.unlink(missing_ok=True)

    with merged.open("wb") as fout:
        for part in parts:
            with part.open("rb") as fin:
                while True:
                    chunk = fin.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
    return merged


def validate_gzip_stream(archive_path: Path, chunk_size: int):
    # If the gzip stream is truncated, gzip.GzipFile will raise an exception
    # before we ever get to tar header parsing.
    total_uncompressed = 0
    with gzip.open(archive_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            total_uncompressed += len(chunk)
    return total_uncompressed


def validate_tar_headers(archive_path: Path):
    # Iterate through tar headers without extracting. A truncated archive will
    # raise tarfile.ReadError or EOF-related exceptions during iteration.
    member_count = 0
    sample_members = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            member_count += 1
            if len(sample_members) < 10:
                sample_members.append(member.name)
    return member_count, sample_members


def main():
    args = parse_args()
    download_dir = Path(args.download_dir).resolve()
    if not download_dir.exists():
        raise FileNotFoundError(f"Download directory not found: {download_dir}")

    chunk_size = args.chunk_size_mb * 1024 * 1024
    parts = list_split_parts(download_dir)
    if not parts:
        raise FileNotFoundError("No CN-Celeb2 split parts found in download_dir")

    print("Found split parts:")
    for part in parts:
        print(f"  {part.name}  size={part.stat().st_size}  md5={md5_of_file(part, chunk_size)}")

    merged = download_dir / "cn-celeb2_v2.tar.gz"
    if args.rebuild or not merged.exists():
        print("Rebuilding merged archive from split parts...")
        merged = rebuild_merged_archive(download_dir, parts)
    else:
        print("Using existing merged archive:")
        print(f"  {merged.name}  size={merged.stat().st_size}  md5={md5_of_file(merged, chunk_size)}")

    print("Validating gzip stream...")
    total_uncompressed = validate_gzip_stream(merged, chunk_size)
    print(f"  gzip readable, total decompressed bytes={total_uncompressed}")

    print("Validating tar headers...")
    member_count, sample_members = validate_tar_headers(merged)
    print(f"  tar readable, member count={member_count}")
    print("  sample members:")
    for name in sample_members:
        print(f"    {name}")

    print("CN-Celeb2 archive check passed")


if __name__ == "__main__":
    main()
