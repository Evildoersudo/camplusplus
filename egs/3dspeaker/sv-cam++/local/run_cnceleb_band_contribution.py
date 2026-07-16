import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

try:
    from speakerlab.process.processor import FBank
    from speakerlab.utils.builder import dynamic_import
    from speakerlab.utils.score_metrics import compute_c_norm, compute_eer, compute_pmiss_pfa_rbst
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from speakerlab.process.processor import FBank
    from speakerlab.utils.builder import dynamic_import
    from speakerlab.utils.score_metrics import compute_c_norm, compute_eer, compute_pmiss_pfa_rbst


CAMPPLUS_COMMON = {
    "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
    "args": {
        "feat_dim": 80,
        "embedding_size": 192,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CN-Celeb band contribution experiment with CAM++ under clean/codec conditions."
    )
    parser.add_argument(
        "--test_wav_scp",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp",
        help="wav.scp under CN-Celeb test split",
    )
    parser.add_argument(
        "--trials_file",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst",
        help="CN-Celeb trials file",
    )
    parser.add_argument(
        "--model_bin",
        type=str,
        default="pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin",
        help="Path to CAM++ pretrained .bin file",
    )
    parser.add_argument(
        "--embedding_cache_root",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/band_contribution/embedding_cache",
        help="Root directory to store cached embedding .npy files",
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/band_contribution/band_contribution_report.csv",
        help="Summary report in CSV",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/band_contribution/band_contribution_report.json",
        help="Summary report in JSON",
    )
    parser.add_argument(
        "--table_md",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/band_contribution/band_contribution_table.md",
        help="Markdown table for paper",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/band_contribution/plots",
        help="Directory to save automatic plots",
    )
    parser.add_argument("--disable_plot", action="store_true", help="Disable automatic plotting")

    parser.add_argument(
        "--codec_conditions",
        type=str,
        default="clean,opus@4k,opus@8k,amrwb@8.85k,amrwb@12.65k",
        help="Comma-separated conditions. Format: clean or codec@bitrate",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="full,low,mid,high,low_mid",
        help="Comma-separated band names to evaluate",
    )
    parser.add_argument("--low_range", type=str, default="1-26", help="1-based mel range for low band")
    parser.add_argument("--mid_range", type=str, default="27-53", help="1-based mel range for mid band")
    parser.add_argument("--high_range", type=str, default="54-80", help="1-based mel range for high band")

    parser.add_argument("--sample_rate", type=int, default=16000, help="Decoded wav sample rate")
    parser.add_argument("--fraction", type=float, default=1.0, help="Use fraction of trials before limit")
    parser.add_argument("--limit", type=int, default=0, help="Use first N (head mode) or sampled N (random mode) trials")
    parser.add_argument(
        "--stratified_sampling",
        action="store_true",
        help="Enable label-stratified trial sampling. Use together with --target_limit and --nontarget_limit for explicit per-class control.",
    )
    parser.add_argument(
        "--trial_sample_mode",
        type=str,
        choices=["head", "random"],
        default="head",
        help="Trial subset policy: head (deterministic prefix) or random (uniform random sampling)",
    )
    parser.add_argument(
        "--trial_sample_seed",
        type=int,
        default=42,
        help="Random seed used when --trial_sample_mode=random",
    )
    parser.add_argument("--overwrite_embeddings", action="store_true", help="Overwrite cached embeddings")
    parser.add_argument("--target_limit", type=int, default=0, help="Optional explicit cap on target trials after stratified sampling")
    parser.add_argument("--nontarget_limit", type=int, default=0, help="Optional explicit cap on nontarget trials after stratified sampling")

    parser.add_argument("--p_target", type=float, default=0.01, help="p_target in minDCF")
    parser.add_argument("--c_miss", type=float, default=1.0, help="c_miss in minDCF")
    parser.add_argument("--c_fa", type=float, default=1.0, help="c_fa in minDCF")
    return parser.parse_args()


def parse_csv_list(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_range(text: str):
    parts = text.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid range: {text}. Expected format like 1-26")
    start = int(parts[0])
    end = int(parts[1])
    if start < 1 or end < start:
        raise ValueError(f"Invalid range: {text}")
    return start, end


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        raise RuntimeError("ffmpeg is not available. Please install ffmpeg and ensure it is in PATH.") from exc


def load_wavscp(wav_scp_path: Path):
    mapping = {}
    with wav_scp_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            utt, path = line.split(maxsplit=1)
            mapping[utt] = Path(path).resolve()
    return mapping


def load_trials(trials_file: Path, fraction: float, limit: int, sample_mode: str, sample_seed: int):
    if not 0 < fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")
    if limit < 0:
        raise ValueError("--limit must be >= 0.")

    lines = [line.strip() for line in trials_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(lines)
    frac_count = total if fraction >= 1.0 else max(1, int(total * fraction))

    if sample_mode == "head":
        selected = lines[:frac_count]
        if limit > 0:
            selected = selected[:limit]
        return selected

    rng = random.Random(sample_seed)
    population = list(range(total))
    target = frac_count
    if limit > 0:
        target = min(target, limit)
    target = min(target, total)
    if target <= 0:
        return []
    chosen_idx = rng.sample(population, target)
    chosen_idx.sort()
    return [lines[i] for i in chosen_idx]


def label_to_binary(label: str):
    return 1 if label in ("1", "target") else 0


def pick_records(records, count: int, sample_mode: str, rng: random.Random):
    if count <= 0 or not records:
        return []
    count = min(count, len(records))
    if sample_mode == "head":
        return records[:count]
    chosen = rng.sample(records, count)
    chosen.sort(key=lambda rec: rec["index"])
    return chosen


def load_trials_stratified(
    trials_file: Path,
    fraction: float,
    limit: int,
    sample_mode: str,
    sample_seed: int,
    target_limit: int,
    nontarget_limit: int,
):
    if not 0 < fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")
    if limit < 0:
        raise ValueError("--limit must be >= 0.")
    if target_limit < 0 or nontarget_limit < 0:
        raise ValueError("--target_limit and --nontarget_limit must be >= 0.")

    raw_lines = [line.strip() for line in trials_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = []
    for idx, line in enumerate(raw_lines):
        parts = line.split()
        if len(parts) != 3:
            continue
        indexed.append({"index": idx, "line": line, "label": parts[2]})

    targets = [rec for rec in indexed if label_to_binary(rec["label"]) == 1]
    nontargets = [rec for rec in indexed if label_to_binary(rec["label"]) == 0]

    if fraction < 1.0:
        target_keep = min(len(targets), max(1, int(len(targets) * fraction))) if targets else 0
        nontarget_keep = min(len(nontargets), max(1, int(len(nontargets) * fraction))) if nontargets else 0
    else:
        target_keep = len(targets)
        nontarget_keep = len(nontargets)

    rng = random.Random(sample_seed)
    frac_targets = pick_records(targets, target_keep, sample_mode, rng)
    frac_nontargets = pick_records(nontargets, nontarget_keep, sample_mode, rng)

    if target_limit > 0 or nontarget_limit > 0:
        chosen_targets = frac_targets if target_limit <= 0 else pick_records(frac_targets, target_limit, sample_mode, rng)
        chosen_nontargets = (
            frac_nontargets if nontarget_limit <= 0 else pick_records(frac_nontargets, nontarget_limit, sample_mode, rng)
        )
        selected = chosen_targets + chosen_nontargets
        if limit > 0 and len(selected) > limit:
            raise ValueError(
                "The sum of stratified class limits exceeds --limit. "
                "Reduce --target_limit/--nontarget_limit or increase --limit."
            )
    else:
        selected = frac_targets + frac_nontargets
        if limit > 0:
            selected = pick_records(selected, limit, sample_mode, rng)

    selected.sort(key=lambda rec: rec["index"])
    return [rec["line"] for rec in selected]


def normalize_trial_utt_key(token: str):
    if token.startswith("enroll-") or token.startswith("test-"):
        return token
    if token.endswith("-enroll"):
        return f"enroll-{token}"
    if "/" in token or "\\" in token or token.lower().endswith(".wav") or token.lower().endswith(".flac"):
        stem = Path(token.replace("\\", "/")).stem
        if stem.startswith("test-"):
            return stem
        return f"test-{stem}"
    return token


def parse_trials_records(trial_lines):
    records = []
    for line in trial_lines:
        utt1_raw, utt2_raw, label = line.split()
        utt1 = normalize_trial_utt_key(utt1_raw)
        utt2 = normalize_trial_utt_key(utt2_raw)
        records.append((utt1, utt2, label, utt1_raw, utt2_raw))
    return records


def extract_utts_from_trials(parsed_trials):
    utts = set()
    for utt1, utt2, _, _, _ in parsed_trials:
        utts.add(utt1)
        utts.add(utt2)
    return sorted(utts)


def parse_condition(text: str):
    if text == "clean":
        return ("clean", "-")
    if "@" not in text:
        raise ValueError(f"Invalid condition: {text}. Use clean or codec@bitrate")
    codec, bitrate = text.split("@", 1)
    codec = codec.strip()
    bitrate = bitrate.strip()
    if not codec or not bitrate:
        raise ValueError(f"Invalid condition: {text}")
    return codec, bitrate


def build_ffmpeg_commands(input_wav: Path, output_wav: Path, codec: str, bitrate: str, sample_rate: int):
    if codec == "opus":
        temp_file = output_wav.with_suffix(".opus")
        encode_cmd = ["ffmpeg", "-y", "-i", str(input_wav), "-c:a", "libopus", "-b:a", bitrate, str(temp_file)]
    elif codec == "g711_mulaw":
        temp_file = output_wav.with_name(output_wav.stem + ".mulaw.wav")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "8000",
            "-ac",
            "1",
            "-c:a",
            "pcm_mulaw",
            str(temp_file),
        ]
    elif codec == "g711_alaw":
        temp_file = output_wav.with_name(output_wav.stem + ".alaw.wav")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "8000",
            "-ac",
            "1",
            "-c:a",
            "pcm_alaw",
            str(temp_file),
        ]
    elif codec == "amrwb":
        temp_file = output_wav.with_suffix(".amr")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "libvo_amrwbenc",
            "-b:a",
            bitrate,
            "-f",
            "amr",
            str(temp_file),
        ]
    elif codec == "aac":
        temp_file = output_wav.with_suffix(".m4a")
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            str(temp_file),
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    decode_cmd = ["ffmpeg", "-y", "-i", str(temp_file), "-ar", str(sample_rate), "-ac", "1", str(output_wav)]
    return temp_file, encode_cmd, decode_cmd


def run_ffmpeg(cmd):
    result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg command failed: {' '.join(cmd)}\n{result.stderr.strip()}")


def run_ffmpeg_capture_bytes(cmd):
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg command failed: {' '.join(cmd)}\n{stderr.strip()}")
    return result.stdout


def load_wav_mono_16k(wav_file: Path, target_sr: int):
    wav, sr = torchaudio.load(str(wav_file))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav[:1]
    return wav


def degrade_to_waveform(clean_wav: Path, codec: str, bitrate: str, sample_rate: int):
    with tempfile.TemporaryDirectory(prefix="band_codec_tmp_") as td:
        # Only keep transient codec artifact in temp dir; do NOT generate
        # persistent degraded wav files used by evaluation.
        tmp_codec = Path(td) / "codec_tmp.wav"
        temp_file, encode_cmd, _ = build_ffmpeg_commands(
            input_wav=clean_wav,
            output_wav=tmp_codec,
            codec=codec,
            bitrate=bitrate,
            sample_rate=sample_rate,
        )
        run_ffmpeg(encode_cmd)

        # Decode compressed audio directly to float PCM in memory.
        decode_pipe_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_file),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "f32le",
            "pipe:1",
        ]
        pcm_bytes = run_ffmpeg_capture_bytes(decode_pipe_cmd)
        pcm = np.frombuffer(pcm_bytes, dtype=np.float32)
        if pcm.size == 0:
            raise RuntimeError(f"Decoded waveform is empty for file: {clean_wav}")
        wav = torch.from_numpy(pcm).unsqueeze(0)

        temp_file.unlink(missing_ok=True)
    return wav


def init_campp_model(model_bin: Path, device: torch.device):
    model_cfg = CAMPPLUS_COMMON
    model = dynamic_import(model_cfg["obj"])(**model_cfg["args"])
    state = torch.load(str(model_bin), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_band_masks(bands, low_range, mid_range, high_range, n_mels=80):
    low_s, low_e = parse_range(low_range)
    mid_s, mid_e = parse_range(mid_range)
    high_s, high_e = parse_range(high_range)

    idx_map = {
        "full": (1, n_mels),
        "low": (low_s, low_e),
        "mid": (mid_s, mid_e),
        "high": (high_s, high_e),
        "low_mid": (low_s, mid_e),
    }

    masks = {}
    for band in bands:
        if band not in idx_map:
            raise ValueError(f"Unsupported band: {band}. Supported: full, low, mid, high, low_mid")
        start, end = idx_map[band]
        if start < 1 or end > n_mels or end < start:
            raise ValueError(f"Band {band} maps to invalid range {start}-{end}")
        mask = torch.zeros(n_mels, dtype=torch.bool)
        mask[start - 1 : end] = True
        masks[band] = mask
    return masks


def embedding_cache_path(cache_root: Path, condition_tag: str, band: str, utt: str):
    safe = utt.replace("/", "_").replace("\\", "_")
    return cache_root / condition_tag / band / f"{safe}.npy"


def compute_embedding_from_wav_with_mask(wav, model, feature_extractor, device: torch.device, band_mask: torch.Tensor):
    feat = feature_extractor(wav)
    feat = feat.clone()
    feat[:, ~band_mask] = 0.0
    feat = feat.unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(feat).detach().squeeze(0).cpu().numpy()
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding is encountered.")
    return vec / norm


def compute_cached_embeddings_for_condition_band(
    utt_to_clean_wav,
    condition_tag: str,
    band: str,
    cache_root: Path,
    model,
    feature_extractor,
    device: torch.device,
    sample_rate: int,
    overwrite_embeddings: bool,
    band_mask: torch.Tensor,
    codec: str = None,
    bitrate: str = None,
):
    emb = {}
    generated = 0
    reused = 0
    utts = sorted(utt_to_clean_wav)

    for idx, utt in enumerate(utts, start=1):
        cache_file = embedding_cache_path(cache_root, condition_tag, band, utt)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if cache_file.exists() and not overwrite_embeddings:
            emb[utt] = np.load(cache_file)
            reused += 1
        else:
            clean_wav = utt_to_clean_wav[utt]
            if codec is None:
                wav = load_wav_mono_16k(clean_wav, sample_rate)
            else:
                wav = degrade_to_waveform(clean_wav, codec, bitrate, sample_rate)
            vec = compute_embedding_from_wav_with_mask(wav, model, feature_extractor, device, band_mask)
            np.save(cache_file, vec)
            emb[utt] = vec
            generated += 1

        if idx % 500 == 0 or idx == len(utts):
            print(
                f"[{condition_tag}][{band}] progress: {idx}/{len(utts)} "
                f"(generated={generated}, reused={reused})"
            )

    return emb


def score_trials(parsed_trials, embeddings):
    labels = []
    scores = []
    for utt1, utt2, label, _, _ in parsed_trials:
        v1 = embeddings[utt1]
        v2 = embeddings[utt2]
        scores.append(float(np.dot(v1, v2)))
        labels.append(label_to_binary(label))
    return np.array(labels), np.array(scores)


def evaluate_metrics(labels: np.ndarray, scores: np.ndarray, p_target: float, c_miss: float, c_fa: float):
    uniq = np.unique(labels)
    if uniq.size < 2:
        raise ValueError(
            "EER/minDCF requires both target and nontarget trials. "
            f"Current subset has labels={uniq.tolist()}. Increase --fraction/--limit."
        )
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, _ = compute_eer(fnr, fpr, scores)
    min_dcf = compute_c_norm(fnr, fpr, p_target=p_target, c_miss=c_miss, c_fa=c_fa)
    return float(100.0 * eer), float(min_dcf)


def enrich_ratios(results):
    baseline = next((r for r in results if r["condition"] == "clean" and r["band"] == "full"), None)
    if baseline is None:
        return results

    clean_full_eer = baseline["eer_percent"]
    clean_full_dcf = baseline["min_dcf"]

    clean_band_map = {r["band"]: r for r in results if r["condition"] == "clean"}

    for r in results:
        if clean_full_eer > 0:
            r["eer_ratio_vs_clean_full"] = r["eer_percent"] / clean_full_eer
        else:
            r["eer_ratio_vs_clean_full"] = None

        if clean_full_dcf > 0:
            r["min_dcf_ratio_vs_clean_full"] = r["min_dcf"] / clean_full_dcf
        else:
            r["min_dcf_ratio_vs_clean_full"] = None

        clean_band = clean_band_map.get(r["band"])
        if clean_band is not None and clean_band["eer_percent"] > 0:
            r["eer_ratio_vs_clean_same_band"] = r["eer_percent"] / clean_band["eer_percent"]
        else:
            r["eer_ratio_vs_clean_same_band"] = None

        if clean_band is not None and clean_band["min_dcf"] > 0:
            r["min_dcf_ratio_vs_clean_same_band"] = r["min_dcf"] / clean_band["min_dcf"]
        else:
            r["min_dcf_ratio_vs_clean_same_band"] = None

    return results


def write_reports(results, report_csv: Path, report_json: Path):
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    with report_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "condition",
                "codec",
                "bitrate",
                "band",
                "num_trials",
                "num_utts",
                "eer_percent",
                "min_dcf",
                "eer_ratio_vs_clean_full",
                "min_dcf_ratio_vs_clean_full",
                "eer_ratio_vs_clean_same_band",
                "min_dcf_ratio_vs_clean_same_band",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["condition"],
                    r["codec"],
                    r["bitrate"],
                    r["band"],
                    r["num_trials"],
                    r["num_utts"],
                    f"{r['eer_percent']:.4f}",
                    f"{r['min_dcf']:.6f}",
                    "" if r["eer_ratio_vs_clean_full"] is None else f"{r['eer_ratio_vs_clean_full']:.4f}",
                    "" if r["min_dcf_ratio_vs_clean_full"] is None else f"{r['min_dcf_ratio_vs_clean_full']:.4f}",
                    "" if r["eer_ratio_vs_clean_same_band"] is None else f"{r['eer_ratio_vs_clean_same_band']:.4f}",
                    ""
                    if r["min_dcf_ratio_vs_clean_same_band"] is None
                    else f"{r['min_dcf_ratio_vs_clean_same_band']:.4f}",
                ]
            )

    report_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


def write_markdown_table(results, table_md: Path, bands, conditions):
    table_md.parent.mkdir(parents=True, exist_ok=True)

    eer_map = {(r["condition"], r["band"]): r["eer_percent"] for r in results}
    dcf_map = {(r["condition"], r["band"]): r["min_dcf"] for r in results}

    lines = ["## 频带贡献实验结果（EER% / minDCF）", ""]
    header = "| 条件 | " + " | ".join(bands) + " |"
    sep = "|---|" + "---|" * len(bands)
    lines.append(header)
    lines.append(sep)
    for cond in conditions:
        vals = []
        for band in bands:
            key = (cond, band)
            if key in eer_map:
                vals.append(f"{eer_map[key]:.3f} / {dcf_map[key]:.4f}")
            else:
                vals.append("-")
        lines.append("| " + cond + " | " + " | ".join(vals) + " |")

    table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric_by_condition(results, bands, conditions, metric_key: str, ylabel: str, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    values = {(r["condition"], r["band"]): r[metric_key] for r in results}
    x = np.arange(len(conditions))
    width = 0.8 / max(1, len(bands))

    plt.figure(figsize=(max(10, len(conditions) * 1.2), 5))
    for idx, band in enumerate(bands):
        series = [values.get((cond, band), np.nan) for cond in conditions]
        offset = (idx - (len(bands) - 1) / 2.0) * width
        plt.bar(x + offset, series, width=width, label=band)

    plt.xticks(x, conditions, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} under different conditions and retained bands")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def write_plots(results, bands, conditions, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    eer_png = plot_dir / "band_contribution_eer.png"
    dcf_png = plot_dir / "band_contribution_min_dcf.png"

    plot_metric_by_condition(
        results=results,
        bands=bands,
        conditions=conditions,
        metric_key="eer_percent",
        ylabel="EER (%)",
        out_png=eer_png,
    )
    plot_metric_by_condition(
        results=results,
        bands=bands,
        conditions=conditions,
        metric_key="min_dcf",
        ylabel="minDCF",
        out_png=dcf_png,
    )
    return eer_png, dcf_png


def main():
    args = parse_args()
    check_ffmpeg()

    test_wav_scp = Path(args.test_wav_scp).resolve()
    trials_file = Path(args.trials_file).resolve()
    model_bin = Path(args.model_bin).resolve()
    embedding_cache_root = Path(args.embedding_cache_root).resolve()
    report_csv = Path(args.report_csv).resolve()
    report_json = Path(args.report_json).resolve()
    table_md = Path(args.table_md).resolve() if args.table_md else None
    plot_dir = Path(args.plot_dir).resolve()

    if not test_wav_scp.exists():
        raise FileNotFoundError(f"test wav.scp not found: {test_wav_scp}")
    if not trials_file.exists():
        raise FileNotFoundError(f"trials file not found: {trials_file}")
    if not model_bin.exists():
        raise FileNotFoundError(f"model bin not found: {model_bin}")

    embedding_cache_root.mkdir(parents=True, exist_ok=True)

    wav_map = load_wavscp(test_wav_scp)
    if args.stratified_sampling:
        trial_lines = load_trials_stratified(
            trials_file=trials_file,
            fraction=args.fraction,
            limit=args.limit,
            sample_mode=args.trial_sample_mode,
            sample_seed=args.trial_sample_seed,
            target_limit=args.target_limit,
            nontarget_limit=args.nontarget_limit,
        )
    else:
        trial_lines = load_trials(
            trials_file=trials_file,
            fraction=args.fraction,
            limit=args.limit,
            sample_mode=args.trial_sample_mode,
            sample_seed=args.trial_sample_seed,
        )
    print(
        f"Trial selection: mode={args.trial_sample_mode}, stratified={args.stratified_sampling}, fraction={args.fraction}, "
        f"limit={args.limit}, seed={args.trial_sample_seed}, selected={len(trial_lines)}"
    )
    parsed_trials = parse_trials_records(trial_lines)
    trial_utts = extract_utts_from_trials(parsed_trials)

    missing = [utt for utt in trial_utts if utt not in wav_map]
    if missing:
        raise RuntimeError(
            f"{len(missing)} utterances in trials are missing in wav.scp after key normalization, first few: {missing[:10]}"
        )

    clean_trial_map = {utt: wav_map[utt] for utt in trial_utts}

    bands = parse_csv_list(args.bands)
    band_masks = build_band_masks(bands, args.low_range, args.mid_range, args.high_range, n_mels=80)

    condition_specs = [parse_condition(x) for x in parse_csv_list(args.codec_conditions)]
    condition_tags = ["clean" if c == "clean" else f"{c}_{b.replace('.', '').replace('/', '_')}" for c, b in condition_specs]

    print("[INFO] This script does NOT generate degraded dataset wav files.")
    print("[INFO] It performs on-the-fly codec simulation -> model forward -> embedding .npy cache -> trial scoring.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = init_campp_model(model_bin, device)
    feature_extractor = FBank(80, sample_rate=args.sample_rate, mean_nor=True)

    results = []

    for codec, bitrate in condition_specs:
        condition_tag = "clean" if codec == "clean" else f"{codec}_{bitrate.replace('.', '').replace('/', '_')}"
        print(f"Running condition: {condition_tag}")
        for band in bands:
            print(f"  Running band: {band}")
            emb = compute_cached_embeddings_for_condition_band(
                utt_to_clean_wav=clean_trial_map,
                condition_tag=condition_tag,
                band=band,
                cache_root=embedding_cache_root,
                model=model,
                feature_extractor=feature_extractor,
                device=device,
                sample_rate=args.sample_rate,
                overwrite_embeddings=args.overwrite_embeddings,
                band_mask=band_masks[band],
                codec=None if codec == "clean" else codec,
                bitrate=None if codec == "clean" else bitrate,
            )
            labels, scores = score_trials(parsed_trials, emb)
            eer, min_dcf = evaluate_metrics(labels, scores, args.p_target, args.c_miss, args.c_fa)
            results.append(
                {
                    "condition": condition_tag,
                    "codec": codec,
                    "bitrate": bitrate,
                    "band": band,
                    "num_trials": len(parsed_trials),
                    "num_utts": len(trial_utts),
                    "eer_percent": eer,
                    "min_dcf": min_dcf,
                }
            )
            print(f"  [{condition_tag}][{band}] EER={eer:.4f}% minDCF={min_dcf:.6f}")

    enrich_ratios(results)
    write_reports(results, report_csv, report_json)
    print(f"Saved CSV report: {report_csv}")
    print(f"Saved JSON report: {report_json}")

    if table_md:
        write_markdown_table(results, table_md, bands, condition_tags)
        print(f"Saved Markdown table: {table_md}")

    if not args.disable_plot:
        eer_png, dcf_png = write_plots(results, bands, condition_tags, plot_dir)
        print(f"Saved EER plot: {eer_png}")
        print(f"Saved minDCF plot: {dcf_png}")


if __name__ == "__main__":
    main()
