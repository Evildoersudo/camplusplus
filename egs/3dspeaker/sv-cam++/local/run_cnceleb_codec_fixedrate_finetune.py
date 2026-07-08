import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a fixed-rate mixed CN-Celeb train set, fine-tune CAM++ from a pretrained checkpoint, and run fixed-rate codec evaluation."
    )
    parser.add_argument(
        "--train_wav_scp",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/wav.scp",
        help="Clean CN-Celeb training wav.scp",
    )
    parser.add_argument(
        "--train_utt2spk",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/utt2spk",
        help="Clean CN-Celeb training utt2spk",
    )
    parser.add_argument(
        "--mixed_train_dir",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed/train",
        help="Output Kaldi-style directory for the fixed-rate mixed train set",
    )
    parser.add_argument(
        "--mixed_audio_root",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed_audio",
        help="Output directory for generated mixed-condition training wavs",
    )
    parser.add_argument(
        "--test_wav_scp",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp",
        help="CN-Celeb test wav.scp used by evaluation",
    )
    parser.add_argument(
        "--trials_file",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst",
        help="CN-Celeb trials file used by evaluation",
    )
    parser.add_argument(
        "--noise_scp",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/musan/wav.scp",
        help="MUSAN wav.scp for training augmentation",
    )
    parser.add_argument(
        "--reverb_scp",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/rirs/wav.scp",
        help="RIRS wav.scp for training augmentation",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="egs/3dspeaker/sv-cam++/conf/cam++_cnceleb_codec16k_ft.yaml",
        help="Training config used for fine-tuning",
    )
    parser.add_argument(
        "--init_model",
        type=str,
        default="pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin",
        help="Pretrained CAM++ embedding checkpoint used to initialize training",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft",
        help="Experiment directory for fine-tuning outputs",
    )
    parser.add_argument(
        "--eval_embedding_cache_root",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/embedding_cache",
        help="Embedding cache directory used by post-training evaluation",
    )
    parser.add_argument(
        "--eval_report_csv",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/report.csv",
        help="Evaluation CSV report path",
    )
    parser.add_argument(
        "--eval_report_json",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/report.json",
        help="Evaluation JSON report path",
    )
    parser.add_argument(
        "--eval_table_md",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/report.md",
        help="Evaluation Markdown table path",
    )
    parser.add_argument(
        "--eval_plot_dir",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/plots",
        help="Evaluation plot directory",
    )
    parser.add_argument(
        "--baseline_embedding_cache_root",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/baseline_embedding_cache",
        help="Embedding cache directory used by pretrained baseline evaluation",
    )
    parser.add_argument(
        "--baseline_report_csv",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/pretrained_report.csv",
        help="Pretrained baseline CSV report path",
    )
    parser.add_argument(
        "--baseline_report_json",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/pretrained_report.json",
        help="Pretrained baseline JSON report path",
    )
    parser.add_argument(
        "--baseline_table_md",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/pretrained_report.md",
        help="Pretrained baseline Markdown table path",
    )
    parser.add_argument(
        "--baseline_plot_dir",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/pretrained_plots",
        help="Pretrained baseline plot directory",
    )
    parser.add_argument(
        "--compare_csv",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/compare_pretrained_vs_finetuned.csv",
        help="Comparison CSV between pretrained and fine-tuned results",
    )
    parser.add_argument(
        "--compare_md",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/compare_pretrained_vs_finetuned.md",
        help="Comparison Markdown table between pretrained and fine-tuned results",
    )
    parser.add_argument(
        "--compare_plot_dir",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/compare_plots",
        help="Comparison plot directory",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-set mixing and evaluation sampling")
    parser.add_argument("--num_workers", type=int, default=8, help="Worker count for mixed train set generation")
    parser.add_argument("--prepare_csv_nj", type=int, default=8, help="Worker count passed to prepare_data_csv.py")
    parser.add_argument("--clean_ratio", type=float, default=0.25, help="Fraction kept clean in mixed train set")
    parser.add_argument("--opus_ratio", type=float, default=0.25, help="Fraction encoded with Opus 16k")
    parser.add_argument("--aac_ratio", type=float, default=0.25, help="Fraction encoded with AAC 16k")
    parser.add_argument("--amrwb_ratio", type=float, default=0.25, help="Fraction encoded with AMR-WB 15.85k")
    parser.add_argument("--g711_ratio", type=float, default=0.0, help="Fraction encoded with G.711")
    parser.add_argument("--opus_bitrates", type=str, default="16k", help="Opus bitrate candidates")
    parser.add_argument("--aac_bitrates", type=str, default="16k", help="AAC bitrate candidates")
    parser.add_argument("--amrwb_bitrates", type=str, default="15.85k", help="AMR-WB bitrate candidates")
    parser.add_argument("--g711_variants", type=str, default="g711_mulaw,g711_alaw", help="G.711 variants")
    parser.add_argument("--gpus", type=str, default="0", help="Space-separated GPU ids, e.g. '0' or '0 1'")
    parser.add_argument("--limit", type=int, default=2000, help="Evaluation total trial cap")
    parser.add_argument("--target_limit", type=int, default=100, help="Evaluation target trial cap")
    parser.add_argument("--nontarget_limit", type=int, default=1900, help="Evaluation nontarget trial cap")
    parser.add_argument("--trial_sample_mode", type=str, choices=["head", "random"], default="random", help="Evaluation trial sampling mode")
    parser.add_argument("--trial_sample_seed", type=int, default=42, help="Evaluation sampling seed")
    parser.add_argument("--overwrite_data", action="store_true", help="Overwrite generated mixed-condition training wavs")
    parser.add_argument("--overwrite_eval_embeddings", action="store_true", help="Overwrite cached evaluation embeddings")
    parser.add_argument("--skip_train", action="store_true", help="Skip fine-tuning and only run evaluation using the latest checkpoint in exp_dir")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation after training")
    return parser.parse_args()


def run_command(cmd, cwd: Path = None):
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def parse_gpus(gpus: str):
    values = [x for x in gpus.split() if x.strip()]
    if not values:
        return ["0"]
    return values


def latest_embedding_ckpt(models_dir: Path):
    ckpt_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for ckpt_dir in ckpt_dirs:
        embedding_ckpt = ckpt_dir / "embedding_model.ckpt"
        if embedding_ckpt.exists():
            return embedding_ckpt
    raise FileNotFoundError(f"No embedding_model.ckpt found under {models_dir}")


def training_num_epochs(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return int(config.get("num_epoch", 0))


def load_eval_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["condition"]: item for item in data}


def write_compare_reports(pretrained_json: Path, finetuned_json: Path, compare_csv: Path, compare_md: Path, compare_plot_dir: Path):
    pretrained = load_eval_json(pretrained_json)
    finetuned = load_eval_json(finetuned_json)
    conditions = [cond for cond in pretrained.keys() if cond in finetuned]

    rows = []
    for condition in conditions:
        pre = pretrained[condition]
        ft = finetuned[condition]
        rows.append(
            {
                "condition": condition,
                "codec": ft["codec"],
                "bitrate": ft["bitrate"],
                "pretrained_eer": pre["eer_percent"],
                "finetuned_eer": ft["eer_percent"],
                "eer_gain": pre["eer_percent"] - ft["eer_percent"],
                "pretrained_min_dcf": pre["min_dcf"],
                "finetuned_min_dcf": ft["min_dcf"],
                "min_dcf_gain": pre["min_dcf"] - ft["min_dcf"],
            }
        )

    compare_csv.parent.mkdir(parents=True, exist_ok=True)
    with compare_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "condition",
                "codec",
                "bitrate",
                "pretrained_eer",
                "finetuned_eer",
                "eer_gain",
                "pretrained_min_dcf",
                "finetuned_min_dcf",
                "min_dcf_gain",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["condition"],
                    row["codec"],
                    row["bitrate"],
                    f"{row['pretrained_eer']:.4f}",
                    f"{row['finetuned_eer']:.4f}",
                    f"{row['eer_gain']:.4f}",
                    f"{row['pretrained_min_dcf']:.6f}",
                    f"{row['finetuned_min_dcf']:.6f}",
                    f"{row['min_dcf_gain']:.6f}",
                ]
            )

    compare_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "## Pretrained vs Fine-tuned CAM++ on fixed-rate codec evaluation",
        "",
        "| condition | codec | bitrate | pretrained EER(%) | finetuned EER(%) | EER gain | pretrained minDCF | finetuned minDCF | minDCF gain |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {codec} | {bitrate} | {pre_eer:.4f} | {ft_eer:.4f} | {eer_gain:.4f} | {pre_dcf:.6f} | {ft_dcf:.6f} | {dcf_gain:.6f} |".format(
                condition=row["condition"],
                codec=row["codec"],
                bitrate=row["bitrate"],
                pre_eer=row["pretrained_eer"],
                ft_eer=row["finetuned_eer"],
                eer_gain=row["eer_gain"],
                pre_dcf=row["pretrained_min_dcf"],
                ft_dcf=row["finetuned_min_dcf"],
                dcf_gain=row["min_dcf_gain"],
            )
        )
    compare_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    compare_plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["condition"] for row in rows]
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))
    plt.bar([i - width / 2 for i in x], [row["pretrained_eer"] for row in rows], width=width, label="pretrained", color="#E45756")
    plt.bar([i + width / 2 for i in x], [row["finetuned_eer"] for row in rows], width=width, label="finetuned", color="#4C78A8")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("EER (%)")
    plt.title("Pretrained vs Fine-tuned EER")
    plt.legend()
    plt.tight_layout()
    plt.savefig(compare_plot_dir / "pretrained_vs_finetuned_eer.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))
    plt.bar([i - width / 2 for i in x], [row["pretrained_min_dcf"] for row in rows], width=width, label="pretrained", color="#E45756")
    plt.bar([i + width / 2 for i in x], [row["finetuned_min_dcf"] for row in rows], width=width, label="finetuned", color="#4C78A8")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("minDCF")
    plt.title("Pretrained vs Fine-tuned minDCF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(compare_plot_dir / "pretrained_vs_finetuned_min_dcf.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))
    gains = [row["eer_gain"] for row in rows]
    plt.bar(labels, gains, color=["#54A24B" if gain >= 0 else "#E45756" for gain in gains])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("EER gain (pretrained - finetuned)")
    plt.title("Fine-tuning gain under each fixed-rate codec condition")
    plt.tight_layout()
    plt.savefig(compare_plot_dir / "pretrained_vs_finetuned_eer_gain.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[4]

    train_wav_scp = Path(args.train_wav_scp).resolve()
    train_utt2spk = Path(args.train_utt2spk).resolve()
    mixed_train_dir = Path(args.mixed_train_dir).resolve()
    mixed_audio_root = Path(args.mixed_audio_root).resolve()
    test_wav_scp = Path(args.test_wav_scp).resolve()
    trials_file = Path(args.trials_file).resolve()
    noise_scp = Path(args.noise_scp).resolve()
    reverb_scp = Path(args.reverb_scp).resolve()
    train_config = Path(args.train_config).resolve()
    init_model = Path(args.init_model).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    eval_embedding_cache_root = Path(args.eval_embedding_cache_root).resolve()
    eval_report_csv = Path(args.eval_report_csv).resolve()
    eval_report_json = Path(args.eval_report_json).resolve()
    eval_table_md = Path(args.eval_table_md).resolve()
    eval_plot_dir = Path(args.eval_plot_dir).resolve()
    baseline_embedding_cache_root = Path(args.baseline_embedding_cache_root).resolve()
    baseline_report_csv = Path(args.baseline_report_csv).resolve()
    baseline_report_json = Path(args.baseline_report_json).resolve()
    baseline_table_md = Path(args.baseline_table_md).resolve()
    baseline_plot_dir = Path(args.baseline_plot_dir).resolve()
    compare_csv = Path(args.compare_csv).resolve()
    compare_md = Path(args.compare_md).resolve()
    compare_plot_dir = Path(args.compare_plot_dir).resolve()

    for path in [train_wav_scp, train_utt2spk, test_wav_scp, trials_file, train_config, init_model]:
        if not path.exists():
            raise FileNotFoundError(path)

    total_ratio = args.clean_ratio + args.opus_ratio + args.aac_ratio + args.amrwb_ratio + args.g711_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("clean_ratio + opus_ratio + aac_ratio + amrwb_ratio + g711_ratio must equal 1.0")
    if args.target_limit + args.nontarget_limit > args.limit:
        raise ValueError("target_limit + nontarget_limit must be <= limit")

    build_script = Path(__file__).resolve().parent / "build_mixed_codec_trainset.py"
    eval_script = Path(__file__).resolve().parent / "run_cnceleb_codec_fixedrate_eval.py"

    baseline_eval_cmd = [
        sys.executable,
        str(eval_script),
        "--test_wav_scp",
        str(test_wav_scp),
        "--trials_file",
        str(trials_file),
        "--model_bin",
        str(init_model),
        "--codec_conditions",
        "clean,opus@16k,aac@16k,amrwb@15.85k",
        "--embedding_cache_root",
        str(baseline_embedding_cache_root),
        "--stratified_sampling",
        "--trial_sample_mode",
        args.trial_sample_mode,
        "--trial_sample_seed",
        str(args.trial_sample_seed),
        "--limit",
        str(args.limit),
        "--target_limit",
        str(args.target_limit),
        "--nontarget_limit",
        str(args.nontarget_limit),
        "--report_csv",
        str(baseline_report_csv),
        "--report_json",
        str(baseline_report_json),
        "--table_md",
        str(baseline_table_md),
        "--plot_dir",
        str(baseline_plot_dir),
    ]
    run_command(baseline_eval_cmd, cwd=project_root)

    build_cmd = [
        sys.executable,
        str(build_script),
        "--input_wav_scp",
        str(train_wav_scp),
        "--input_utt2spk",
        str(train_utt2spk),
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
        "--aac_ratio",
        str(args.aac_ratio),
        "--amrwb_ratio",
        str(args.amrwb_ratio),
        "--g711_ratio",
        str(args.g711_ratio),
        "--opus_bitrates",
        args.opus_bitrates,
        "--aac_bitrates",
        args.aac_bitrates,
        "--amrwb_bitrates",
        args.amrwb_bitrates,
        "--g711_variants",
        args.g711_variants,
    ]
    if args.overwrite_data:
        build_cmd.append("--overwrite")
    run_command(build_cmd, cwd=project_root)

    train_csv = mixed_train_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Mixed train.csv not found: {train_csv}")

    if not args.skip_train:
        gpu_list = parse_gpus(args.gpus)
        if len(gpu_list) <= 1:
            train_cmd = [
                sys.executable,
                "-m",
                "speakerlab.bin.train",
                "--config",
                str(train_config),
                "--gpu",
                *gpu_list,
                "--init_model",
                str(init_model),
                "--data",
                str(train_csv),
                "--noise",
                str(noise_scp),
                "--reverb",
                str(reverb_scp),
                "--exp_dir",
                str(exp_dir),
            ]
            run_command(train_cmd, cwd=project_root)
        else:
            train_cmd = [
                "torchrun",
                f"--nproc_per_node={len(gpu_list)}",
                "-m",
                "speakerlab.bin.train",
                "--config",
                str(train_config),
                "--gpu",
                *gpu_list,
                "--init_model",
                str(init_model),
                "--data",
                str(train_csv),
                "--noise",
                str(noise_scp),
                "--reverb",
                str(reverb_scp),
                "--exp_dir",
                str(exp_dir),
            ]
            run_command(train_cmd, cwd=project_root)

    if args.skip_eval:
        return

    models_dir = exp_dir / "models"
    model_bin = latest_embedding_ckpt(models_dir)

    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--test_wav_scp",
        str(test_wav_scp),
        "--trials_file",
        str(trials_file),
        "--model_bin",
        str(model_bin),
        "--codec_conditions",
        "clean,opus@16k,aac@16k,amrwb@15.85k",
        "--embedding_cache_root",
        str(eval_embedding_cache_root),
        "--stratified_sampling",
        "--trial_sample_mode",
        args.trial_sample_mode,
        "--trial_sample_seed",
        str(args.trial_sample_seed),
        "--limit",
        str(args.limit),
        "--target_limit",
        str(args.target_limit),
        "--nontarget_limit",
        str(args.nontarget_limit),
        "--report_csv",
        str(eval_report_csv),
        "--report_json",
        str(eval_report_json),
        "--table_md",
        str(eval_table_md),
        "--plot_dir",
        str(eval_plot_dir),
    ]
    if args.overwrite_eval_embeddings:
        eval_cmd.append("--overwrite_embeddings")
    run_command(eval_cmd, cwd=project_root)

    write_compare_reports(
        pretrained_json=baseline_report_json,
        finetuned_json=eval_report_json,
        compare_csv=compare_csv,
        compare_md=compare_md,
        compare_plot_dir=compare_plot_dir,
    )

    print("Fixed-rate codec fine-tuning and evaluation finished")
    print(f"Train config: {train_config}")
    print(f"Training epochs: {training_num_epochs(train_config)}")
    print(f"Fine-tuned embedding checkpoint: {model_bin}")
    print(f"Evaluation JSON: {eval_report_json}")
    print(f"Baseline JSON: {baseline_report_json}")
    print(f"Comparison CSV: {compare_csv}")


if __name__ == "__main__":
    main()
