import argparse
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_args():
    # This script prepares a CN-Celeb recipe layout from the downloaded
    # archives, then builds a mixed codec training set for CAM++ fine-tuning.
    """
    解析命令行参数。
    该脚本的主要任务是从下载的压缩包中准备 CN-Celeb 数据集的布局，
    并为 CAM++ 模型微调构建一个混合编码（Mixed Codec）的训练集。
    """
    parser = argparse.ArgumentParser(
        description="Prepare CN-Celeb clean/mixed train data, test metadata, and MUSAN/RIRS metadata."
    )
    # --- 基础路径配置 ---
    parser.add_argument("--download_dir", type=str, required=True, help="Directory containing downloaded archives")
    parser.add_argument("--data_root", type=str, default="data", help="Recipe data root")
    parser.add_argument(
        "--workspace_name",
        type=str,
        default="CN_celeb_database",
        help="Subdirectory under data_root used to store all generated CN-Celeb metadata and mixed data",
    )
    
    parser.add_argument("--raw_root", type=str, default="", help="Optional extracted raw data root")
    # --- 音频处理与并发配置 ---
    parser.add_argument("--sample_rate", type=int, default=16000, help="Expected training sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mixed split")
    parser.add_argument("--num_workers", type=int, default=8, help="Worker count for codec generation")
    parser.add_argument("--prepare_csv_nj", type=int, default=8, help="Worker count for prepare_data_csv.py")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite extracted files and generated audio")
    # --- 混合编码比例配置 (各类音质在训练集中的占比) ---
    parser.add_argument("--max_utts", type=int, default=0, help="Optional utterance cap for debugging")
    parser.add_argument("--clean_ratio", type=float, default=0.30, help="Fraction kept clean in mixed train set")
    parser.add_argument("--opus_ratio", type=float, default=0.30, help="Fraction encoded with Opus")
    parser.add_argument("--amrwb_ratio", type=float, default=0.20, help="Fraction encoded with AMR-WB")
    parser.add_argument("--aac_ratio", type=float, default=0.00, help="Fraction encoded with AAC")
    parser.add_argument("--g711_ratio", type=float, default=0.20, help="Fraction encoded with G.711")
    # --- 编码器特定参数 (码率和变体) ---
    parser.add_argument("--opus_bitrates", type=str, default="4k,6k,8k", help="Opus bitrate candidates")
    parser.add_argument(
        "--amrwb_bitrates",
        type=str,
        default="8.85k,8.85k,8.85k,12.65k,23.85k",
        help="AMR-WB bitrate candidates. Repeat 8.85k to give it more weight.",
    )
    parser.add_argument("--aac_bitrates", type=str, default="16k", help="AAC bitrate candidates")
    parser.add_argument(
        "--g711_variants",
        type=str,
        default="g711_mulaw,g711_alaw",
        help="G.711 variants used in the mixed train set",
    )
    return parser.parse_args()


def run_command(cmd):
    """通用辅助函数：执行 Shell 命令，如果失败则抛出异常停止程序"""
    subprocess.run(cmd, check=True)


def ensure_cnceleb2_archive(download_dir: Path):
    """
    确保 CN-Celeb2 的压缩包是完整的。
    由于官方数据很大，通常分卷下载（.tar.gz, .tar.gzaa, .tar.gzab 等）。
    此函数会检测分卷文件，并将它们合并成一个完整的 cn-celeb2_v2.tar.gz 文件。
    """
    archive = download_dir / "cn-celeb2_v2.tar.gz"
    # 如果完整的压缩包已经存在，直接返回它的路径
    if archive.exists():
        return archive
    # 扫描目录下所有的分卷文件并排序
    parts = sorted(download_dir.glob("cn-celeb2_v2.tar.gz*"))
    if not parts:
        return None
    # 以二进制追加写入的模式，把所有分卷拼接起来
    with archive.open("wb") as fout:
        for part in parts:
            with part.open("rb") as fin:
                shutil.copyfileobj(fin, fout)
    return archive


def rebuild_cnceleb2_archive(download_dir: Path):
    """
    当之前的合并文件丢失或损坏导致解压失败时，调用此函数。
    它会先删除损坏的压缩包，然后重新执行合并逻辑。
    """
    # Rebuild the merged archive from split parts when the previous merged file
    # is missing or corrupted.
    archive = download_dir / "cn-celeb2_v2.tar.gz"
    archive.unlink(missing_ok=True)
    return ensure_cnceleb2_archive(download_dir)


def extract_if_needed(download_dir: Path, raw_root: Path):
    """
    核心解压模块：检查目标文件夹是否存在，不存在才进行解压。
    防止脚本中断后重新运行导致重复耗时。
    """
    raw_root.mkdir(parents=True, exist_ok=True)

    musan_archive = download_dir / "musan.tar.gz"
    rirs_archive = download_dir / "rirs_noises.zip"
    cnceleb1_archive = download_dir / "cn-celeb_v2.tar.gz"
    cnceleb2_archive = ensure_cnceleb2_archive(download_dir)

    if not (raw_root / "musan").exists():
        run_command(["tar", "-xzf", str(musan_archive), "-C", str(raw_root)])

    if not (raw_root / "RIRS_NOISES").exists():
        with zipfile.ZipFile(rirs_archive, "r") as zf:
            zf.extractall(raw_root)

    if not (raw_root / "CN-Celeb_flac").exists():
        run_command(["tar", "-xzf", str(cnceleb1_archive), "-C", str(raw_root)])

    if cnceleb2_archive is not None and not (raw_root / "CN-Celeb2_flac").exists():
        try:
            run_command(["tar", "-xzf", str(cnceleb2_archive), "-C", str(raw_root)])
        except subprocess.CalledProcessError as exc:
            # A truncated archive usually means the merged tar.gz was built
            # from incomplete parts or an old partial file. Remove both the
            # half-extracted directory and the merged archive so the next run
            # rebuilds from the split parts.
            shutil.rmtree(raw_root / "CN-Celeb2_flac", ignore_errors=True)
            (download_dir / "cn-celeb2_v2.tar.gz").unlink(missing_ok=True)
            raise RuntimeError(
                "CN-Celeb2 解压失败，当前合并后的 cn-celeb2_v2.tar.gz 很可能已损坏或不完整。"
                " 请确认 cn-celeb2_v2.tar.gzaa / .gzab / .gzac 都下载完整，然后重新运行脚本。"
            ) from exc


def utt_id_from_path(path: Path, speaker_id: str):
    """
    根据文件路径生成全局唯一的语音 ID (Utterance ID)。
    为了防止不同说话人的文件名冲突，统一格式化为: {speaker_id}-{文件名}
    """
    stem = path.stem
    if stem.startswith(f"{speaker_id}-") or stem.startswith(f"{speaker_id}_"):
        return stem
    return f"{speaker_id}-{stem}"


def write_spk2utt(utt2spk_path: Path, spk2utt_path: Path):
    """
    Kaldi 数据格式转换：根据 utt2spk 生成反向映射的 spk2utt。
    输入格式: [语音ID] [说话人ID]
    输出格式: [说话人ID] [语音ID_1] [语音ID_2] ...
    """
    spk2utt = defaultdict(list)
    with utt2spk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            utt_id, spk_id = line.strip().split()
            spk2utt[spk_id].append(utt_id)

    with spk2utt_path.open("w", encoding="utf-8", newline="\n") as fout:
        for spk_id in sorted(spk2utt):
            fout.write(f"{spk_id} {' '.join(sorted(spk2utt[spk_id]))}\n")


def build_split_metadata(file_paths, split_dir: Path):
    """
    构建 Kaldi 风格的元数据文件 (wav.scp 和 utt2spk)。
    这是语音模型训练的数据字典基础。
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_scp_path = split_dir / "wav.scp"
    utt2spk_path = split_dir / "utt2spk"

    with wav_scp_path.open("w", encoding="utf-8", newline="\n") as fwav, utt2spk_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as futt:
        for wav_path in sorted(file_paths):
            speaker_id = wav_path.parent.name
            utt_id = utt_id_from_path(wav_path, speaker_id)
            # 写入音频绝对路径映射
            fwav.write(f"{utt_id} {wav_path.resolve()}\n")#{speaker_id}-{文件名}-{文件绝对路径}
            # 写入语音到说话人的映射
            futt.write(f"{utt_id} {speaker_id}\n")
    # 生成对应的反向映射表
    write_spk2utt(utt2spk_path, split_dir / "spk2utt")


def build_noise_metadata(raw_root: Path, workspace_root: Path):
    """
    构建背景噪声 (MUSAN) 和房间混响 (RIRS) 的映射字典 (wav.scp)。
    用于后续训练时的数据增强 (Data Augmentation)。
    """
    musan_dir = workspace_root / "musan"
    rirs_dir = workspace_root / "rirs"
    musan_dir.mkdir(parents=True, exist_ok=True)
    rirs_dir.mkdir(parents=True, exist_ok=True)

    noise_files = sorted((raw_root / "musan" / "noise" / "free-sound").rglob("*.wav"))
    with (musan_dir / "wav.scp").open("w", encoding="utf-8", newline="\n") as fout:
        for wav_path in noise_files:
            rel = wav_path.relative_to(raw_root / "musan")
            utt_id = str(rel).replace("\\", "/")
            fout.write(f"{utt_id} {wav_path.resolve()}\n")
# 读取 RIRS 官方提供的 rir_list，解析并转换为本地绝对路径
    rir_list = raw_root / "RIRS_NOISES" / "real_rirs_isotropic_noises" / "rir_list"
    with (rirs_dir / "wav.scp").open("w", encoding="utf-8", newline="\n") as fout:
        with rir_list.open("r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                rel_path = parts[4]
                wav_path = (raw_root / rel_path).resolve()
                fout.write(f"{rel_path} {wav_path}\n")


def copy_trial_files(raw_root: Path, trials_dir: Path):
    """
    提取官方测试用例 (Trials)。
    这些文件记录了用于测试模型的“考题”（例如：判断音频A和音频B是否为同一人）。
    """
    trials_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = raw_root / "CN-Celeb_flac" / "eval"

    # Keep this generic because CN-Celeb releases are not always packaged with
    # exactly the same auxiliary file names.
    candidates = []
    for path in eval_dir.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if "trial" in lower_name or "pairs" in lower_name or lower_name.endswith(".lst"):
            candidates.append(path)

    for src in sorted(candidates):
        shutil.copy2(src, trials_dir / src.name)


def prepare_clean_and_test(raw_root: Path, workspace_root: Path, sample_rate: int, prepare_csv_nj: int):
    cnceleb_root = workspace_root / "cnceleb"
    clean_train_dir = cnceleb_root / "clean_train"
    test_dir = cnceleb_root / "test"
    trials_dir = cnceleb_root / "trials"
    # 汇总 CN-Celeb1 和 CN-Celeb2 的训练音频
    cnceleb1_train = sorted((raw_root / "CN-Celeb_flac" / "data").rglob("*.flac"))
    cnceleb2_root = raw_root / "CN-Celeb2_flac" / "data"
    if cnceleb2_root.exists():
        cnceleb2_train = sorted(cnceleb2_root.rglob("*.flac"))
    else:
        cnceleb2_train = []
    train_files = cnceleb1_train + cnceleb2_train
    if not train_files:
        raise ValueError("No CN-Celeb training audio found after extraction")

    # Use all official eval audio under eval/ as the test pool so trials can
    # reference both enrol and test utterances if present.
    # 收集所有的评估音频 (eval)
    eval_audio = []
    eval_root = raw_root / "CN-Celeb_flac" / "eval"
    for suffix in ("*.flac", "*.wav"):
        eval_audio.extend(eval_root.rglob(suffix))
    if not eval_audio:
        raise ValueError("No CN-Celeb eval audio found under CN-Celeb_flac/eval")
    # 构建基础元数据文本
    build_split_metadata(train_files, clean_train_dir)
    build_split_metadata(eval_audio, test_dir)
    copy_trial_files(raw_root, trials_dir)
    # 外包任务：调用同目录下的外部脚本 prepare_data_csv.py，将元数据转为 CSV 格式
    prepare_csv_script = Path(__file__).resolve().parent / "prepare_data_csv.py"
    run_command(
        [
            sys.executable,
            str(prepare_csv_script),
            "--data_dir",
            str(clean_train_dir),
            "--sample_rate",
            str(sample_rate),
            "--nj",
            str(prepare_csv_nj),
        ]
    )


def build_mixed_trainset(args, workspace_root: Path):
    clean_train_dir = workspace_root / "cnceleb" / "clean_train"
    mixed_train_dir = workspace_root / "cnceleb_mixed" / "train"
    mixed_audio_root = workspace_root / "cnceleb_mixed_audio"

    script_path = Path(__file__).resolve().parent / "build_mixed_codec_trainset.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--input_wav_scp",
        str(clean_train_dir / "wav.scp"),
        "--input_utt2spk",
        str(clean_train_dir / "utt2spk"),
        "--output_dir",
        str(mixed_train_dir),
        "--audio_output_root",
        str(mixed_audio_root),
        "--sample_rate",
        str(args.sample_rate),
        "--seed",
        str(args.seed),
        "--num_workers",
        str(args.num_workers),
        "--prepare_csv_nj",
        str(args.prepare_csv_nj),
        "--clean_ratio",
        str(args.clean_ratio),
        "--opus_ratio",
        str(args.opus_ratio),
        "--amrwb_ratio",
        str(args.amrwb_ratio),
        "--aac_ratio",
        str(args.aac_ratio),
        "--g711_ratio",
        str(args.g711_ratio),
        "--opus_bitrates",
        args.opus_bitrates,
        "--amrwb_bitrates",
        args.amrwb_bitrates,
        "--aac_bitrates",
        args.aac_bitrates,
        "--g711_variants",
        args.g711_variants,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.max_utts > 0:
        cmd.extend(["--max_utts", str(args.max_utts)])
    run_command(cmd)


def main():
    args = parse_args()
    # 将路径全部转换为绝对路径 (resolve)，防止执行命令时找不到文件
    data_root = Path(args.data_root).resolve()
    workspace_root = (data_root / args.workspace_name).resolve()
    download_dir = Path(args.download_dir).resolve()
    raw_root = Path(args.raw_root).resolve() if args.raw_root else (data_root / "raw_data")
    # 步骤 1：解压原始压缩包
    extract_if_needed(download_dir, raw_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    # 步骤 2：准备噪声数据元信息
    build_noise_metadata(raw_root, workspace_root)
    # 步骤 3：处理主要语音数据并转换 CSV
    prepare_clean_and_test(raw_root, workspace_root, args.sample_rate, args.prepare_csv_nj)
    # 步骤 4：生成混合编码数据（高强度数据增强）
    build_mixed_trainset(args, workspace_root)

    print("CN-Celeb mixed data preparation finished")
    print(f"Raw root: {raw_root}")
    print(f"Workspace root: {workspace_root}")
    print(f"Clean train dir: {workspace_root / 'cnceleb' / 'clean_train'}")
    print(f"Mixed train dir: {workspace_root / 'cnceleb_mixed' / 'train'}")
    print(f"Test dir: {workspace_root / 'cnceleb' / 'test'}")
    print(f"Trials dir: {workspace_root / 'cnceleb' / 'trials'}")
    print(f"MUSAN wav.scp: {workspace_root / 'musan' / 'wav.scp'}")
    print(f"RIRS wav.scp: {workspace_root / 'rirs' / 'wav.scp'}")


if __name__ == "__main__":
    main()
