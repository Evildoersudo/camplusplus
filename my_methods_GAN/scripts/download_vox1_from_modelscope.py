import argparse
import os
import shutil
from pathlib import Path

from modelscope import snapshot_download
#下载 LibriSpeech，默认只下载 train-clean-100.tar.gz
# python my_methods_GAN/scripts/download_vox1_from_modelscope.py \
#   --dataset librispeech

#下载 train-other-500.tar.gz
# python my_methods_GAN/scripts/download_vox1_from_modelscope.py \
#  --dataset librispeech \
#  --librispeech-parts train-clean-100 train-other-500



VOX_DATASET_ID = "juliuscn/voxceleb"
LIBRISPEECH_DATASET_ID = "pkufool/LibriSpeech"

# 这是你图片里显示的真实远端目录
VOX_REMOTE_DIR = "vox1"

# 你希望最终保存的位置
VOX_TARGET_DIR = Path("/home/dgx/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data/vox1")
LIBRISPEECH_TARGET_DIR = Path("/home/dgx/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data")

# 临时下载目录，用来接收ModelScope按仓库结构下载下来的文件
VOX_TMP_DIR = Path("/home/dgx/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data/_modelscope_voxceleb_tmp")
LIBRISPEECH_TMP_DIR = Path("/home/dgx/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/_modelscope_librispeech_tmp")

LIBRISPEECH_PARTS = {
    "train-clean-100": "train-clean-100.tar.gz",
    "train-other-500": "train-other-500.tar.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VoxCeleb1 or selected LibriSpeech archives from ModelScope."
    )
    parser.add_argument(
        "--dataset",
        choices=("vox1", "librispeech"),
        default="vox1",
        help="Dataset to download. Defaults to the original VoxCeleb1 behavior.",
    )
    parser.add_argument(
        "--librispeech-parts",
        nargs="+",
        choices=sorted(LIBRISPEECH_PARTS),
        default=["train-clean-100"],
        help=(
            "LibriSpeech archives to download. train-other-500 is optional "
            "and is not downloaded unless explicitly listed."
        ),
    )
    return parser.parse_args()


def configure_cache() -> None:
    # 尽量把缓存也放到autodl-tmp，避免占用/root系统盘
    os.environ["MODELSCOPE_CACHE"] = "/home/dgx/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data/modelscope_cache"


def print_target_files(target_dir: Path) -> None:
    print("\nCurrent files:")
    for p in sorted(target_dir.iterdir()):
        size_gb = p.stat().st_size / 1024 / 1024 / 1024 if p.is_file() else 0
        if p.is_file():
            print(f"  {p.name}\t{size_gb:.2f} GB")
        else:
            print(f"  {p.name}/")


def download_vox1() -> None:
    VOX_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    VOX_TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Dataset: {VOX_DATASET_ID}")
    print(f"Remote dir: {VOX_REMOTE_DIR}")
    print(f"Temp download dir: {VOX_TMP_DIR}")
    print(f"Final target dir: {VOX_TARGET_DIR}")
    print("=" * 80)

    # 下载voxceleb/vox1目录下的全部文件
    local_repo_dir = snapshot_download(
        repo_id=VOX_DATASET_ID,
        repo_type="dataset",
        local_dir=str(VOX_TMP_DIR),
        allow_patterns=[
            f"{VOX_REMOTE_DIR}/*",
        ],
    )

    print(f"\nSnapshot downloaded to: {local_repo_dir}")

    src_dir = VOX_TMP_DIR / VOX_REMOTE_DIR

    if not src_dir.exists():
        raise FileNotFoundError(
            f"Expected downloaded directory not found: {src_dir}\n"
            f"Please check whether the remote directory is really {VOX_REMOTE_DIR}"
        )

    # 将TMP_DIR/voxceleb/vox1/下的文件移动到/root/autodl-tmp/raw_data/vox1/
    for src_path in src_dir.iterdir():
        dst_path = VOX_TARGET_DIR / src_path.name

        if dst_path.exists():
            print(f"[Skip] Already exists: {dst_path}")
            continue

        print(f"[Move] {src_path} -> {dst_path}")
        shutil.move(str(src_path), str(dst_path))

    print("\nDownload finished.")
    print(f"Files are saved in: {VOX_TARGET_DIR}")
    print_target_files(VOX_TARGET_DIR)


def download_librispeech(parts: list[str]) -> None:
    LIBRISPEECH_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    LIBRISPEECH_TMP_DIR.mkdir(parents=True, exist_ok=True)

    filenames = [LIBRISPEECH_PARTS[part] for part in parts]

    print("=" * 80)
    print(f"Dataset: {LIBRISPEECH_DATASET_ID}")
    print(f"Archives: {', '.join(filenames)}")
    print(f"Temp download dir: {LIBRISPEECH_TMP_DIR}")
    print(f"Final target dir: {LIBRISPEECH_TARGET_DIR}")
    print("=" * 80)

    local_repo_dir = snapshot_download(
        repo_id=LIBRISPEECH_DATASET_ID,
        repo_type="dataset",
        local_dir=str(LIBRISPEECH_TMP_DIR),
        allow_patterns=filenames,
    )

    print(f"\nSnapshot downloaded to: {local_repo_dir}")

    for filename in filenames:
        src_path = LIBRISPEECH_TMP_DIR / filename
        dst_path = LIBRISPEECH_TARGET_DIR / filename

        if not src_path.is_file() and dst_path.is_file():
            print(f"[Skip] Already exists: {dst_path}")
            continue
        if not src_path.is_file():
            raise FileNotFoundError(
                f"Expected downloaded archive not found: {src_path}"
            )
        if dst_path.exists():
            print(f"[Skip] Already exists: {dst_path}")
            continue

        print(f"[Move] {src_path} -> {dst_path}")
        shutil.move(str(src_path), str(dst_path))

    print("\nDownload finished.")
    print(f"Files are saved in: {LIBRISPEECH_TARGET_DIR}")
    print_target_files(LIBRISPEECH_TARGET_DIR)


def main():
    args = parse_args()
    configure_cache()

    if args.dataset == "vox1":
        download_vox1()
    else:
        download_librispeech(args.librispeech_parts)


if __name__ == "__main__":
    main()
