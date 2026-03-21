import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        description="Evaluate CAM++ on CN-Celeb test utterances under fixed-rate codec conditions and export paper-ready tables/plots."
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
        "--codec_conditions",
        type=str,
        default="clean,opus@16k,aac@16k,amrwb@15.85k",
        help="Comma-separated conditions. Format: clean or codec@bitrate. Default keeps codecs that have an exact or near-16k mode.",
    )
    parser.add_argument(
        "--embedding_cache_root",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/embedding_cache",
        help="Root directory to store cached embedding .npy files",
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/codec_fixedrate_report.csv",
        help="Summary report in CSV",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/codec_fixedrate_report.json",
        help="Summary report in JSON",
    )
    parser.add_argument(
        "--table_md",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/codec_fixedrate_table.md",
        help="Markdown table for paper",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/plots",
        help="Directory to save automatic plots",
    )
    parser.add_argument("--disable_plot", action="store_true", help="Disable automatic plotting")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Decoded wav sample rate")
    parser.add_argument("--fraction", type=float, default=1.0, help="Use fraction of trials before other sampling")
    parser.add_argument("--limit", type=int, default=0, help="Use first N trials in head mode or sampled N trials in random mode")
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
        help="Trial subset policy: head or random",
    )
    parser.add_argument(
        "--trial_sample_seed",
        type=int,
        default=42,
        help="Random seed used by random trial/utterance sampling",
    )
    parser.add_argument(
        "--max_utts",
        type=int,
        default=0,
        help="Maximum number of distinct test utterances to keep after trial selection. 0 means keep all.",
    )
    parser.add_argument(
        "--utt_sample_mode",
        type=str,
        choices=["head", "random"],
        default="random",
        help="How to downsample distinct test utterances when --max_utts > 0",
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
        records.append(
            {
                "utt1": normalize_trial_utt_key(utt1_raw),
                "utt2": normalize_trial_utt_key(utt2_raw),
                "label": label,
                "utt1_raw": utt1_raw,
                "utt2_raw": utt2_raw,
            }
        )
    return records


def label_to_binary(label: str):
    return 1 if label in ("1", "target") else 0


def summarize_trial_labels(records):
    target = sum(label_to_binary(rec["label"]) for rec in records)
    nontarget = len(records) - target
    return target, nontarget


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

    if target_limit > 0:
        frac_targets = pick_records(frac_targets, target_limit, sample_mode, rng)
    if nontarget_limit > 0:
        frac_nontargets = pick_records(frac_nontargets, nontarget_limit, sample_mode, rng)

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


def sample_trial_records_by_utt(records, max_utts: int, sample_mode: str, sample_seed: int, random_retries: int = 50):
    if max_utts <= 0:
        return records

    utts = sorted({rec["utt1"] for rec in records} | {rec["utt2"] for rec in records})
    if max_utts >= len(utts):
        return records

    if sample_mode == "head":
        chosen = set(utts[:max_utts])
        filtered = [rec for rec in records if rec["utt1"] in chosen and rec["utt2"] in chosen]
        if not filtered:
            raise ValueError("No trials remain after applying --max_utts. Increase the value or disable utterance sampling.")
        return filtered

    rng = random.Random(sample_seed)
    best_filtered = []
    best_has_both_labels = False
    best_target = 0

    for _ in range(max(1, random_retries)):
        chosen = set(rng.sample(utts, max_utts))
        filtered = [rec for rec in records if rec["utt1"] in chosen and rec["utt2"] in chosen]
        if not filtered:
            continue

        target, nontarget = summarize_trial_labels(filtered)
        has_both_labels = target > 0 and nontarget > 0

        if has_both_labels:
            if (not best_has_both_labels) or len(filtered) > len(best_filtered):
                best_filtered = filtered
                best_has_both_labels = True
                best_target = target
        elif not best_has_both_labels and len(filtered) > len(best_filtered):
            best_filtered = filtered
            best_target = target

    if not best_filtered:
        raise ValueError("No trials remain after applying --max_utts. Increase the value or disable utterance sampling.")

    best_nontarget = len(best_filtered) - best_target
    if best_target <= 0 or best_nontarget <= 0:
        raise ValueError(
            "After applying --max_utts, sampled trials still contain only one class. "
            "Increase --max_utts, disable utterance sampling, or increase --limit."
        )
    return best_filtered


def build_balanced_trial_subset(records, max_utts: int, sample_mode: str, sample_seed: int, random_retries: int = 50):
    if max_utts <= 0:
        return records

    utts = sorted({rec["utt1"] for rec in records} | {rec["utt2"] for rec in records})
    if max_utts >= len(utts):
        return records

    # Keep the same primary trial-sampling behavior as run_cnceleb_band_contribution.py.
    # Only apply this second-stage utterance cap when the user explicitly asks for it.
    return sample_trial_records_by_utt(
        records=records,
        max_utts=max_utts,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        random_retries=random_retries,
    )


def extract_utts_from_trials(parsed_trials):
    utts = set()
    for rec in parsed_trials:
        utts.add(rec["utt1"])
        utts.add(rec["utt2"])
    return sorted(utts)


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


def load_wav_mono_16k(wav_file: Path, target_sr: int):
    wav, sr = torchaudio.load(str(wav_file))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav[:1]
    return wav


def degrade_to_waveform(clean_wav: Path, codec: str, bitrate: str, sample_rate: int):
    with tempfile.TemporaryDirectory(prefix="codec_fixedrate_tmp_") as td:
        tmp_decoded = Path(td) / "decoded.wav"
        _, encode_cmd, decode_cmd = build_ffmpeg_commands(
            input_wav=clean_wav,
            output_wav=tmp_decoded,
            codec=codec,
            bitrate=bitrate,
            sample_rate=sample_rate,
        )
        run_ffmpeg(encode_cmd)
        run_ffmpeg(decode_cmd)
        wav = load_wav_mono_16k(tmp_decoded, sample_rate)
    return wav


def embedding_cache_path(cache_root: Path, condition_tag: str, utt: str):
    safe = utt.replace("/", "_").replace("\\", "_")
    return cache_root / condition_tag / f"{safe}.npy"


def compute_embedding_from_wav(wav: torch.Tensor, model, feature_extractor, device: torch.device):
    feat = feature_extractor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(feat).detach().squeeze(0).cpu().numpy()
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding is encountered.")
    return vec / norm


def compute_cached_embeddings(
    utt_to_clean_wav,
    condition_tag: str,
    cache_root: Path,
    model,
    feature_extractor,
    device: torch.device,
    sample_rate: int,
    overwrite_embeddings: bool,
    codec: str = None,
    bitrate: str = None,
):
    emb = {}
    generated = 0
    reused = 0
    utts = sorted(utt_to_clean_wav)

    for idx, utt in enumerate(utts, start=1):
        cache_file = embedding_cache_path(cache_root, condition_tag, utt)
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
            vec = compute_embedding_from_wav(wav, model, feature_extractor, device)
            np.save(cache_file, vec)
            emb[utt] = vec
            generated += 1

        if idx % 500 == 0 or idx == len(utts):
            print(
                f"[{condition_tag}] embedding progress: {idx}/{len(utts)} "
                f"(generated={generated}, reused={reused})"
            )

    return emb


def init_campp_model(model_bin: Path, device: torch.device):
    model = dynamic_import(CAMPPLUS_COMMON["obj"])(**CAMPPLUS_COMMON["args"])
    state = torch.load(str(model_bin), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def score_trials(parsed_trials, embeddings):
    labels = []
    scores = []
    for rec in parsed_trials:
        v1 = embeddings[rec["utt1"]]
        v2 = embeddings[rec["utt2"]]
        scores.append(float(np.dot(v1, v2)))
        labels.append(label_to_binary(rec["label"]))
    return np.array(labels), np.array(scores)


def evaluate_metrics(labels: np.ndarray, scores: np.ndarray, p_target: float, c_miss: float, c_fa: float):
    uniq = np.unique(labels)
    if uniq.size < 2:
        raise ValueError(
            "EER/minDCF requires both target and nontarget trials. "
            f"Current subset has labels={uniq.tolist()}. Increase --fraction/--limit or --max_utts."
        )
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, _ = compute_eer(fnr, fpr, scores)
    min_dcf = compute_c_norm(fnr, fpr, p_target=p_target, c_miss=c_miss, c_fa=c_fa)
    return float(100.0 * eer), float(min_dcf)


def condition_to_tag(codec: str, bitrate: str):
    return "clean" if codec == "clean" else f"{codec}_{bitrate.replace('.', '').replace('/', '_')}"


def enrich_results(results):
    clean = next((item for item in results if item["condition"] == "clean"), None)
    if clean is None:
        return results

    clean_eer = clean["eer_percent"]
    clean_min_dcf = clean["min_dcf"]

    for item in results:
        item["eer_delta_vs_clean"] = item["eer_percent"] - clean_eer
        item["min_dcf_delta_vs_clean"] = item["min_dcf"] - clean_min_dcf
        item["eer_ratio_vs_clean"] = None if clean_eer <= 0 else item["eer_percent"] / clean_eer
        item["min_dcf_ratio_vs_clean"] = None if clean_min_dcf <= 0 else item["min_dcf"] / clean_min_dcf
    return results


def write_reports(results, csv_path: Path, json_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "condition",
                "codec",
                "bitrate",
                "num_trials",
                "num_utts",
                "eer_percent",
                "min_dcf",
                "eer_delta_vs_clean",
                "min_dcf_delta_vs_clean",
                "eer_ratio_vs_clean",
                "min_dcf_ratio_vs_clean",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item["condition"],
                    item["codec"],
                    item["bitrate"],
                    item["num_trials"],
                    item["num_utts"],
                    f"{item['eer_percent']:.4f}",
                    f"{item['min_dcf']:.6f}",
                    f"{item['eer_delta_vs_clean']:.4f}",
                    f"{item['min_dcf_delta_vs_clean']:.6f}",
                    "" if item["eer_ratio_vs_clean"] is None else f"{item['eer_ratio_vs_clean']:.4f}",
                    "" if item["min_dcf_ratio_vs_clean"] is None else f"{item['min_dcf_ratio_vs_clean']:.4f}",
                ]
            )

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def write_markdown_table(results, table_md: Path):
    table_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "## CN-Celeb fixed-rate codec evaluation",
        "",
        "| condition | codec | bitrate | #trials | #utts | EER(%) | minDCF | Delta EER | Delta minDCF | EER/Clean | minDCF/Clean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {condition} | {codec} | {bitrate} | {num_trials} | {num_utts} | {eer:.4f} | {dcf:.6f} | {eer_delta:.4f} | {dcf_delta:.6f} | {eer_ratio} | {dcf_ratio} |".format(
                condition=item["condition"],
                codec=item["codec"],
                bitrate=item["bitrate"],
                num_trials=item["num_trials"],
                num_utts=item["num_utts"],
                eer=item["eer_percent"],
                dcf=item["min_dcf"],
                eer_delta=item["eer_delta_vs_clean"],
                dcf_delta=item["min_dcf_delta_vs_clean"],
                eer_ratio="" if item["eer_ratio_vs_clean"] is None else f"{item['eer_ratio_vs_clean']:.4f}",
                dcf_ratio="" if item["min_dcf_ratio_vs_clean"] is None else f"{item['min_dcf_ratio_vs_clean']:.4f}",
            )
        )
    table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_absolute_metrics(results, metric_key: str, ylabel: str, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    labels = [item["condition"] for item in results]
    values = [item[metric_key] for item in results]
    colors = ["#4C78A8" if item["condition"] == "clean" else "#E45756" for item in results]

    plt.figure(figsize=(max(8, len(labels) * 1.4), 5))
    bars = plt.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} under fixed-rate codec conditions")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_relative_metrics(results, metric_key: str, ylabel: str, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    degraded = [item for item in results if item["condition"] != "clean"]
    degraded = [item for item in degraded if item[metric_key] is not None]
    if not degraded:
        return

    labels = [item["condition"] for item in degraded]
    values = [item[metric_key] for item in degraded]

    plt.figure(figsize=(max(8, len(labels) * 1.4), 5))
    bars = plt.bar(labels, values, color="#72B7B2")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.2f}x", ha="center", va="bottom", fontsize=9)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} relative to clean")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def write_plots(results, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    eer_png = plot_dir / "fixedrate_eer.png"
    dcf_png = plot_dir / "fixedrate_min_dcf.png"
    rel_png = plot_dir / "fixedrate_relative_eer.png"

    plot_absolute_metrics(results, "eer_percent", "EER (%)", eer_png)
    plot_absolute_metrics(results, "min_dcf", "minDCF", dcf_png)
    plot_relative_metrics(results, "eer_ratio_vs_clean", "EER / Clean", rel_png)
    return eer_png, dcf_png, rel_png


def main():
    args = parse_args()
    check_ffmpeg()

    test_wav_scp = Path(args.test_wav_scp).resolve()
    trials_file = Path(args.trials_file).resolve()
    model_bin = Path(args.model_bin).resolve()
    cache_root = Path(args.embedding_cache_root).resolve()
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

    condition_specs = [parse_condition(item) for item in parse_csv_list(args.codec_conditions)]
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
    parsed_trials = parse_trials_records(trial_lines)
    parsed_trials = build_balanced_trial_subset(
        records=parsed_trials,
        max_utts=args.max_utts,
        sample_mode=args.utt_sample_mode,
        sample_seed=args.trial_sample_seed,
    )
    trial_utts = extract_utts_from_trials(parsed_trials)
    num_target, num_nontarget = summarize_trial_labels(parsed_trials)

    print(
        f"Trial selection: mode={args.trial_sample_mode}, stratified={args.stratified_sampling}, fraction={args.fraction}, "
        f"limit={args.limit}, seed={args.trial_sample_seed}, selected_trials={len(parsed_trials)}, "
        f"selected_utts={len(trial_utts)}, target_trials={num_target}, nontarget_trials={num_nontarget}"
    )

    missing = [utt for utt in trial_utts if utt not in wav_map]
    if missing:
        raise RuntimeError(
            f"{len(missing)} utterances in trials are missing in wav.scp after key normalization, first few: {missing[:10]}"
        )

    clean_trial_map = {utt: wav_map[utt] for utt in trial_utts}
    cache_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = init_campp_model(model_bin, device)
    feature_extractor = FBank(80, sample_rate=args.sample_rate, mean_nor=True)

    results = []
    for codec, bitrate in condition_specs:
        condition_tag = condition_to_tag(codec, bitrate)
        print(f"Running condition: {condition_tag}")
        embeddings = compute_cached_embeddings(
            utt_to_clean_wav=clean_trial_map,
            condition_tag=condition_tag,
            cache_root=cache_root,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            sample_rate=args.sample_rate,
            overwrite_embeddings=args.overwrite_embeddings,
            codec=None if codec == "clean" else codec,
            bitrate=None if codec == "clean" else bitrate,
        )
        labels, scores = score_trials(parsed_trials, embeddings)
        eer, min_dcf = evaluate_metrics(labels, scores, args.p_target, args.c_miss, args.c_fa)
        results.append(
            {
                "condition": condition_tag,
                "codec": codec,
                "bitrate": bitrate,
                "num_trials": len(parsed_trials),
                "num_utts": len(trial_utts),
                "eer_percent": eer,
                "min_dcf": min_dcf,
            }
        )
        print(f"[{condition_tag}] EER={eer:.4f}% minDCF={min_dcf:.6f}")

    enrich_results(results)
    write_reports(results, report_csv, report_json)
    print(f"Saved CSV report: {report_csv}")
    print(f"Saved JSON report: {report_json}")

    if table_md:
        write_markdown_table(results, table_md)
        print(f"Saved Markdown table: {table_md}")

    if not args.disable_plot:
        eer_png, dcf_png, rel_png = write_plots(results, plot_dir)
        print(f"Saved EER plot: {eer_png}")
        print(f"Saved minDCF plot: {dcf_png}")
        print(f"Saved relative-EER plot: {rel_png}")


if __name__ == "__main__":
    main()
