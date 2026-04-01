"""Evaluate CA-AFC frontend + frozen CAM++ under fixed-rate codec conditions.

This script mirrors the earlier CN-Celeb codec evaluation style, but inserts the
learned CA-AFC frontend before CAM++ embedding extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_methods.data.ca_afc_data import compute_aux_features, compute_fbank, load_frozen_campplus, load_wav_mono
from my_methods.models.ca_afc_frontend import CAAFCFrontend
from speakerlab.utils.score_metrics import compute_c_norm, compute_eer, compute_pmiss_pfa_rbst


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run CA-AFC + CAM++ fixed-rate codec evaluation on CN-Celeb trials.")
    parser.add_argument("--test_wav_scp", type=str, required=True, help="Test wav.scp")
    parser.add_argument("--trials_file", type=str, required=True, help="Trial list file")
    parser.add_argument("--frontend_ckpt", type=str, required=True, help="CA-AFC checkpoint saved by train_ca_afc.py")
    parser.add_argument("--campplus_model_bin", type=str, required=True, help="Pretrained CAM++ checkpoint")
    parser.add_argument("--codec_conditions", type=str, default="clean,opus@16k,aac@16k,amrwb@15.85k", help="Comma-separated codec conditions")
    parser.add_argument("--embedding_cache_root", type=str, required=True, help="Directory for cached embeddings")
    parser.add_argument("--report_csv", type=str, required=True, help="Output CSV report")
    parser.add_argument("--report_json", type=str, required=True, help="Output JSON report")
    parser.add_argument("--table_md", type=str, required=True, help="Output Markdown table")
    parser.add_argument("--plot_dir", type=str, required=True, help="Output plot directory")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Evaluation sample rate")
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device")
    parser.add_argument("--overwrite_embeddings", action="store_true", help="Recompute cached embeddings")
    parser.add_argument("--limit", type=int, default=2000, help="Total sampled trials")
    parser.add_argument("--target_limit", type=int, default=100, help="Target sampled trials")
    parser.add_argument("--nontarget_limit", type=int, default=1900, help="Nontarget sampled trials")
    parser.add_argument("--trial_sample_seed", type=int, default=42, help="Random seed for trial sampling")
    parser.add_argument("--p_target", type=float, default=0.01, help="Target prior for minDCF")
    parser.add_argument("--c_miss", type=float, default=1.0, help="Miss cost for minDCF")
    parser.add_argument("--c_fa", type=float, default=1.0, help="False alarm cost for minDCF")
    return parser.parse_args()


def parse_condition(tag: str):
    """Parse `clean` or `codec@bitrate` into `(codec, bitrate, condition_tag)`."""

    if tag == "clean":
        return "clean", "-", "clean"
    codec, bitrate = tag.split("@", 1)
    condition_tag = f"{codec}_{bitrate.replace('.', '').replace('/', '_')}"
    return codec, bitrate, condition_tag


def resolve_runtime_audio_path(path_like: str | Path) -> Path:
    """Resolve host/container wav paths to a readable runtime path."""

    path = Path(path_like)
    if path.exists():
        return path.resolve()

    parts = PurePosixPath(str(path_like)).parts
    if "camplusplus" in parts:
        anchor = parts.index("camplusplus")
        suffix_parts = parts[anchor + 1 :]
        remapped = REPO_ROOT.joinpath(*suffix_parts)
        if remapped.exists():
            return remapped.resolve()

    return path.resolve()


def load_wav_scp(path: Path):
    """Load wav.scp into a `utt -> wav_path` mapping."""

    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            utt, wav = line.split(maxsplit=1)
            mapping[utt] = resolve_runtime_audio_path(wav)
    return mapping


def label_to_binary(label: str) -> int:
    """Normalize common trial labels to binary target flags."""

    value = str(label).strip().lower()
    if value in {"1", "target", "true"}:
        return 1
    if value in {"0", "nontarget", "non-target", "false"}:
        return 0
    raise ValueError(f"Unsupported trial label: {label}")


def normalize_trial_utt(utt: str) -> str:
    """Normalize CN-Celeb trial utterance ids to match wav.scp keys."""

    value = str(utt).strip().replace("\\", "/")
    if not value:
        raise ValueError("Encountered empty utterance id in trials.")
    if value.startswith(("enroll-", "test-")):
        return value
    if value.endswith("-enroll"):
        return f"enroll-{value}"
    if value.startswith("test/"):
        stem = Path(value).stem
        return f"test-{stem}"
    return value


def load_trials(path: Path):
    """Load all trials as `(label, utt1, utt2)` tuples."""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            first, second, third = parts
            try:
                label = label_to_binary(first)
                utt1 = normalize_trial_utt(second)
                utt2 = normalize_trial_utt(third)
            except ValueError:
                label = label_to_binary(third)
                utt1 = normalize_trial_utt(first)
                utt2 = normalize_trial_utt(second)
            rows.append((label, utt1, utt2))
    return rows


def sample_trials_stratified(trials, limit: int, target_limit: int, nontarget_limit: int, seed: int):
    """Sample a reproducible stratified subset for stable CN-Celeb reporting."""

    if target_limit + nontarget_limit > limit:
        raise ValueError("target_limit + nontarget_limit must be <= limit")

    rng = np.random.default_rng(seed)
    targets = [item for item in trials if item[0] == 1]
    nontargets = [item for item in trials if item[0] == 0]

    rng.shuffle(targets)
    rng.shuffle(nontargets)
    picked_targets = targets[:target_limit]
    picked_nontargets = nontargets[:nontarget_limit]
    picked = picked_targets + picked_nontargets
    rng.shuffle(picked)
    return picked[:limit]


def build_ffmpeg_commands(input_wav: Path, output_wav: Path, codec: str, bitrate: str, sample_rate: int):
    """Create encode/decode commands for supported codec conditions."""

    if codec == "opus":
        temp_file = output_wav.with_suffix(".opus")
        encode_cmd = ["ffmpeg", "-y", "-i", str(input_wav), "-ar", "16000", "-ac", "1", "-c:a", "libopus", "-b:a", bitrate, str(temp_file)]
    elif codec == "aac":
        temp_file = output_wav.with_suffix(".m4a")
        encode_cmd = ["ffmpeg", "-y", "-i", str(input_wav), "-ar", "16000", "-ac", "1", "-c:a", "aac", "-b:a", bitrate, str(temp_file)]
    elif codec == "amrwb":
        temp_file = output_wav.with_suffix(".amr")
        encode_cmd = ["ffmpeg", "-y", "-i", str(input_wav), "-ar", "16000", "-ac", "1", "-c:a", "libvo_amrwbenc", "-b:a", bitrate, "-f", "amr", str(temp_file)]
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    decode_cmd = ["ffmpeg", "-y", "-i", str(temp_file), "-ar", str(sample_rate), "-ac", "1", str(output_wav)]
    return encode_cmd, decode_cmd


def run_ffmpeg(cmd):
    """Run an ffmpeg command and surface stderr on failure."""

    result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {' '.join(cmd)}\n{result.stderr.strip()}")


def degrade_to_waveform(clean_wav: Path, codec: str, bitrate: str, sample_rate: int):
    """Encode/decode one waveform on the fly for evaluation."""

    with tempfile.TemporaryDirectory(prefix="ca_afc_eval_") as td:
        decoded = Path(td) / "decoded.wav"
        encode_cmd, decode_cmd = build_ffmpeg_commands(clean_wav, decoded, codec, bitrate, sample_rate)
        run_ffmpeg(encode_cmd)
        run_ffmpeg(decode_cmd)
        return load_wav_mono(decoded, sample_rate)


def load_frontend(frontend_ckpt: Path, device: torch.device):
    """Load CA-AFC frontend from a training checkpoint."""

    state = torch.load(str(frontend_ckpt), map_location="cpu")
    args = state.get("args", {})
    codec_vocab_text = str(args.get("codec_vocab", "clean,aac,opus,amrwb,g711,unknown"))
    codec_vocab = []
    seen = set()
    for item in codec_vocab_text.split(","):
        name = item.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        codec_vocab.append(name)
    if "unknown" not in seen:
        codec_vocab.append("unknown")

    frontend = CAAFCFrontend(
        hidden_dim=int(args.get("hidden_dim", 64)),
        dropout=float(args.get("dropout", 0.1)),
        band_scale=float(args.get("band_scale", 1.0)),
        residual_scale=float(args.get("residual_scale", 0.1)),
        num_codecs=len(codec_vocab),
        codec_emb_dim=int(args.get("codec_emb_dim", 16)),
    )
    frontend.load_state_dict(state["frontend_state"])
    frontend.eval()
    frontend.to(device)
    frontend.codec_to_id = {name: idx for idx, name in enumerate(codec_vocab)}
    return frontend


def embedding_cache_path(cache_root: Path, condition_tag: str, utt: str):
    """Map one utterance/condition pair to its embedding cache path."""

    safe = utt.replace("/", "_").replace("\\", "_")
    return cache_root / condition_tag / f"{safe}.npy"


def extract_embedding_from_wav(
    wav: torch.Tensor,
    codec_name: str,
    frontend,
    campplus,
    sample_rate: int,
    device: torch.device,
):
    """Run CA-AFC then CAM++ and return a unit-normalized embedding."""

    codec_feat = compute_fbank(wav, sample_rate).unsqueeze(0).to(device)
    aux_feat = compute_aux_features(wav, sample_rate).unsqueeze(0).to(device)
    min_len = min(codec_feat.shape[1], aux_feat.shape[1])
    codec_feat = codec_feat[:, :min_len]
    aux_feat = aux_feat[:, :min_len]

    codec_to_id = getattr(frontend, "codec_to_id", {"unknown": 0})
    codec_id = codec_to_id.get(str(codec_name).lower(), codec_to_id.get("unknown", 0))
    codec_ids = torch.tensor([int(codec_id)], dtype=torch.long, device=device)

    with torch.no_grad():
        enhanced = frontend(codec_feat, aux_feat, codec_ids=codec_ids).enhanced
        embedding = campplus(enhanced).squeeze(0).cpu().numpy()

    norm = np.linalg.norm(embedding)
    if norm <= 0:
        raise RuntimeError("Encountered zero-norm embedding.")
    return embedding / norm


def compute_embeddings_for_condition(utt_to_wav, needed_utts, cache_root: Path, condition_tag: str, codec: str, bitrate: str, frontend, campplus, args, device):
    """Compute or reuse embeddings for all utterances under one codec condition."""

    generated = 0
    reused = 0
    for idx, utt in enumerate(sorted(needed_utts), start=1):
        cache_path = embedding_cache_path(cache_root, condition_tag, utt)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not args.overwrite_embeddings:
            reused += 1
            continue

        wav = load_wav_mono(utt_to_wav[utt], args.sample_rate) if codec == "clean" else degrade_to_waveform(utt_to_wav[utt], codec, bitrate, args.sample_rate)
        embedding = extract_embedding_from_wav(wav, codec, frontend, campplus, args.sample_rate, device)
        np.save(cache_path, embedding)
        generated += 1

        if idx % 50 == 0 or idx == len(needed_utts):
            print(f"[{condition_tag}] embedding progress: {idx}/{len(needed_utts)} (generated={generated}, reused={reused})")


def evaluate_metrics(labels, scores, p_target: float, c_miss: float, c_fa: float):
    """Compute EER and minDCF from trial labels and cosine scores."""

    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, _ = compute_eer(fnr, fpr, scores)
    min_dcf = compute_c_norm(fnr, fpr, p_target=p_target, c_miss=c_miss, c_fa=c_fa)
    return float(100.0 * eer), float(min_dcf)


def write_reports(results, report_csv: Path, report_json: Path, table_md: Path):
    """Write CSV, JSON, and Markdown summaries."""

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "codec", "bitrate", "num_trials", "num_utts", "eer_percent", "min_dcf"])
        for item in results:
            writer.writerow([item["condition"], item["codec"], item["bitrate"], item["num_trials"], item["num_utts"], f"{item['eer_percent']:.4f}", f"{item['min_dcf']:.6f}"])

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "## CA-AFC + CAM++ fixed-rate codec evaluation",
        "",
        "| condition | codec | bitrate | trials | utts | EER(%) | minDCF |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['condition']} | {item['codec']} | {item['bitrate']} | {item['num_trials']} | {item['num_utts']} | {item['eer_percent']:.4f} | {item['min_dcf']:.6f} |"
        )
    table_md.parent.mkdir(parents=True, exist_ok=True)
    table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(results, plot_dir: Path):
    """Create visual summaries for EER and minDCF."""

    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [item["condition"] for item in results]

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))
    plt.bar(labels, [item["eer_percent"] for item in results], color="#4C78A8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("EER (%)")
    plt.title("CA-AFC + CAM++ fixed-rate EER")
    plt.tight_layout()
    plt.savefig(plot_dir / "ca_afc_eer.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(8, len(labels) * 1.5), 5))
    plt.bar(labels, [item["min_dcf"] for item in results], color="#E45756")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("minDCF")
    plt.title("CA-AFC + CAM++ fixed-rate minDCF")
    plt.tight_layout()
    plt.savefig(plot_dir / "ca_afc_min_dcf.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    utt_to_wav = load_wav_scp(Path(args.test_wav_scp).resolve())
    trials = load_trials(Path(args.trials_file).resolve())
    trials = sample_trials_stratified(trials, args.limit, args.target_limit, args.nontarget_limit, args.trial_sample_seed)
    needed_utts = {utt for _, utt1, utt2 in trials for utt in (utt1, utt2)}
    missing_utts = sorted(utt for utt in needed_utts if utt not in utt_to_wav)
    if missing_utts:
        preview = ", ".join(missing_utts[:10])
        raise KeyError(f"{len(missing_utts)} trial utterances are missing from wav.scp. First few: {preview}")
    print(f"Trial selection: selected_trials={len(trials)}, selected_utts={len(needed_utts)}")

    frontend = load_frontend(Path(args.frontend_ckpt).resolve(), device=device)
    campplus = load_frozen_campplus(Path(args.campplus_model_bin).resolve(), device=device)

    results = []
    cache_root = Path(args.embedding_cache_root).resolve()
    for tag in [item.strip() for item in args.codec_conditions.split(",") if item.strip()]:
        codec, bitrate, condition_tag = parse_condition(tag)
        compute_embeddings_for_condition(
            utt_to_wav=utt_to_wav,
            needed_utts=needed_utts,
            cache_root=cache_root,
            condition_tag=condition_tag,
            codec=codec,
            bitrate=bitrate,
            frontend=frontend,
            campplus=campplus,
            args=args,
            device=device,
        )

        labels = []
        scores = []
        for label, utt1, utt2 in trials:
            emb1 = np.load(embedding_cache_path(cache_root, condition_tag, utt1))
            emb2 = np.load(embedding_cache_path(cache_root, condition_tag, utt2))
            labels.append(label)
            scores.append(float(np.dot(emb1, emb2)))

        eer, min_dcf = evaluate_metrics(np.array(labels), np.array(scores), args.p_target, args.c_miss, args.c_fa)
        results.append(
            {
                "condition": tag,
                "codec": codec,
                "bitrate": bitrate,
                "num_trials": len(trials),
                "num_utts": len(needed_utts),
                "eer_percent": eer,
                "min_dcf": min_dcf,
            }
        )
        print(f"[{tag}] EER={eer:.4f}% minDCF={min_dcf:.6f}")

    write_reports(results, Path(args.report_csv).resolve(), Path(args.report_json).resolve(), Path(args.table_md).resolve())
    write_plots(results, Path(args.plot_dir).resolve())
    print("CA-AFC codec evaluation finished.")


if __name__ == "__main__":
    main()
