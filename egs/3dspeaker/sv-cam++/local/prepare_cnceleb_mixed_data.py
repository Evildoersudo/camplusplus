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
    parser = argparse.ArgumentParser(
        description="Prepare CN-Celeb clean/mixed train data, test metadata, and MUSAN/RIRS metadata."
    )
    parser.add_argument("--download_dir", type=str, required=True, help="Directory containing downloaded archives")
    parser.add_argument("--data_root", type=str, default="data", help="Recipe data root")
    parser.add_argument(
        "--workspace_name",
        type=str,
        default="CN_celeb_database",
        help="Subdirectory under data_root used to store all generated CN-Celeb metadata and mixed data",
    )
    parser.add_argument("--raw_root", type=str, default="", help="Optional extracted raw data root")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Expected training sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mixed split")
    parser.add_argument("--num_workers", type=int, default=8, help="Worker count for codec generation")
    parser.add_argument("--prepare_csv_nj", type=int, default=8, help="Worker count for prepare_data_csv.py")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite extracted files and generated audio")
    parser.add_argument("--max_utts", type=int, default=0, help="Optional utterance cap for debugging")
    parser.add_argument("--clean_ratio", type=float, default=0.30, help="Fraction kept clean in mixed train set")
    parser.add_argument("--opus_ratio", type=float, default=0.30, help="Fraction encoded with Opus")
    parser.add_argument("--amrwb_ratio", type=float, default=0.20, help="Fraction encoded with AMR-WB")
    parser.add_argument("--g711_ratio", type=float, default=0.20, help="Fraction encoded with G.711")
    parser.add_argument("--opus_bitrates", type=str, default="4k,6k,8k", help="Opus bitrate candidates")
    parser.add_argument(
        "--amrwb_bitrates",
        type=str,
        default="8.85k,8.85k,8.85k,12.65k,23.85k",
        help="AMR-WB bitrate candidates. Repeat 8.85k to give it more weight.",
    )
    parser.add_argument(
        "--g711_variants",
        type=str,
        default="g711_mulaw,g711_alaw",
        help="G.711 variants used in the mixed train set",
    )
    return parser.parse_args()


def run_command(cmd):
    subprocess.run(cmd, check=True)


def ensure_cnceleb2_archive(download_dir: Path):
    archive = download_dir / "cn-celeb2_v2.tar.gz"
    if archive.exists():
        return archive

    parts = sorted(download_dir.glob("cn-celeb2_v2.tar.gz*"))
    if not parts:
        return None

    with archive.open("wb") as fout:
        for part in parts:
            with part.open("rb") as fin:
                shutil.copyfileobj(fin, fout)
    return archive


def rebuild_cnceleb2_archive(download_dir: Path):
    # Rebuild the merged archive from split parts when the previous merged file
    # is missing or corrupted.
    archive = download_dir / "cn-celeb2_v2.tar.gz"
    archive.unlink(missing_ok=True)
    return ensure_cnceleb2_archive(download_dir)


def extract_if_needed(download_dir: Path, raw_root: Path):
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
    stem = path.stem
    if stem.startswith(f"{speaker_id}-") or stem.startswith(f"{speaker_id}_"):
        return stem
    return f"{speaker_id}-{stem}"


def write_spk2utt(utt2spk_path: Path, spk2utt_path: Path):
    spk2utt = defaultdict(list)
    with utt2spk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            utt_id, spk_id = line.strip().split()
            spk2utt[spk_id].append(utt_id)

    with spk2utt_path.open("w", encoding="utf-8", newline="\n") as fout:
        for spk_id in sorted(spk2utt):
            fout.write(f"{spk_id} {' '.join(sorted(spk2utt[spk_id]))}\n")


def build_split_metadata(file_paths, split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_scp_path = split_dir / "wav.scp"
    utt2spk_path = split_dir / "utt2spk"

    with wav_scp_path.open("w", encoding="utf-8", newline="\n") as fwav, utt2spk_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as futt:
        for wav_path in sorted(file_paths):
            speaker_id = wav_path.parent.name
            utt_id = utt_id_from_path(wav_path, speaker_id)
            fwav.write(f"{utt_id} {wav_path.resolve()}\n")
            futt.write(f"{utt_id} {speaker_id}\n")

    write_spk2utt(utt2spk_path, split_dir / "spk2utt")


def build_noise_metadata(raw_root: Path, workspace_root: Path):
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
    eval_audio = []
    eval_root = raw_root / "CN-Celeb_flac" / "eval"
    for suffix in ("*.flac", "*.wav"):
        eval_audio.extend(eval_root.rglob(suffix))
    if not eval_audio:
        raise ValueError("No CN-Celeb eval audio found under CN-Celeb_flac/eval")

    build_split_metadata(train_files, clean_train_dir)
    build_split_metadata(eval_audio, test_dir)
    copy_trial_files(raw_root, trials_dir)

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
        "--g711_ratio",
        str(args.g711_ratio),
        "--opus_bitrates",
        args.opus_bitrates,
        "--amrwb_bitrates",
        args.amrwb_bitrates,
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
    data_root = Path(args.data_root).resolve()
    workspace_root = (data_root / args.workspace_name).resolve()
    download_dir = Path(args.download_dir).resolve()
    raw_root = Path(args.raw_root).resolve() if args.raw_root else (data_root / "raw_data")

    extract_if_needed(download_dir, raw_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    build_noise_metadata(raw_root, workspace_root)
    prepare_clean_and_test(raw_root, workspace_root, args.sample_rate, args.prepare_csv_nj)
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
