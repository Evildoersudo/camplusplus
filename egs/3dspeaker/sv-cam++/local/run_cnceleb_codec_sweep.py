import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
import tempfile

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
        description="One-stop CN-Celeb codec sweep: on-the-fly codec simulation, embedding cache, and CAM++ EER/minDCF."
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
        "--degraded_root",
        type=str,
        default="egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_codec_sweep",
        help="Deprecated compatibility arg; no persistent degraded wavs are generated in cache mode.",
    )
    parser.add_argument(
        "--embedding_cache_root",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/embedding_cache",
        help="Root directory to store cached embedding .npy files",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/cnceleb_codec_sweep_report.json",
        help="Summary report in JSON",
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/cnceleb_codec_sweep_report.csv",
        help="Summary report in CSV",
    )
    parser.add_argument(
        "--ranking_csv",
        type=str,
        default="",
        help="Optional grouped ranking CSV for paper tables",
    )
    parser.add_argument(
        "--ranking_md",
        type=str,
        default="",
        help="Optional grouped ranking Markdown table for paper tables",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Decoded wav sample rate")
    parser.add_argument("--num_workers", type=int, default=8, help="Reserved for future parallel codec processing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached embeddings for all conditions")
    parser.add_argument("--overwrite_embeddings", action="store_true", help="Overwrite cached embeddings")
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
    parser.add_argument(#随机种子
        "--trial_sample_seed",
        type=int,
        default=42,
        help="Random seed used when --trial_sample_mode=random",
    )

    parser.add_argument("--target_limit", type=int, default=0, help="Optional explicit cap on target trials after stratified sampling")
    parser.add_argument("--nontarget_limit", type=int, default=0, help="Optional explicit cap on nontarget trials after stratified sampling")

    parser.add_argument("--opus_bitrates", type=str, default="4k,8k,16k", help="Comma-separated Opus bitrates")
    parser.add_argument(
        "--amrwb_bitrates",
        type=str,
        default="8.85k,12.65k,23.85k",
        help="Comma-separated AMR-WB bitrates",
    )
    parser.add_argument("--aac_bitrates", type=str, default="16k,32k,64k", help="Comma-separated AAC bitrates")
    parser.add_argument(
        "--g711_variants",
        type=str,
        default="g711_mulaw,g711_alaw",
        help="Comma-separated G.711 variants",
    )

    parser.add_argument("--p_target", type=float, default=0.01, help="p_target in minDCF")
    parser.add_argument("--c_miss", type=float, default=1.0, help="c_miss in minDCF")
    parser.add_argument("--c_fa", type=float, default=1.0, help="c_fa in minDCF")
    return parser.parse_args()


def parse_csv_list(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


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
    # CN-Celeb trials can be in forms like:
    # 1) id00800-enroll
    # 2) test/id00800-speech-01-001.wav
    # while wav.scp keys are usually:
    # enroll-id00800-enroll / test-id00800-speech-01-001
    if token.startswith("enroll-") or token.startswith("test-"):
        return token

    if token.endswith("-enroll"):
        return f"enroll-{token}"

    if "/" in token or "\\" in token or token.lower().endswith(".wav") or token.lower().endswith(".flac"):
        stem = Path(token.replace("\\", "/")).stem
        if stem.startswith("test-"):
            return stem
        return f"test-{stem}"

    # Keep original token for datasets where trials already match wav.scp keys.
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


def embedding_cache_path(cache_root: Path, condition_tag: str, utt: str):
    safe = utt.replace("/", "_").replace("\\", "_")
    return cache_root / condition_tag / f"{safe}.npy"


def compute_embedding_from_wav(
    wav: torch.Tensor,
    model,
    feature_extractor,
    device: torch.device,
):
    feat = feature_extractor(wav).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(feat).detach().squeeze(0).cpu().numpy()
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding is encountered.")
    return vec / norm


def degrade_to_waveform(clean_wav: Path, codec: str, bitrate: str, sample_rate: int):
    with tempfile.TemporaryDirectory(prefix="codec_tmp_") as td:
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
    cache_dir = cache_root / condition_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    emb = {}
    reused = 0
    generated = 0

    utts = sorted(utt_to_clean_wav)
    for idx, utt in enumerate(utts, start=1):
        cache_file = embedding_cache_path(cache_root, condition_tag, utt)
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
    model_cfg = CAMPPLUS_COMMON
    model = dynamic_import(model_cfg["obj"])(**model_cfg["args"])
    state = torch.load(str(model_bin), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


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


def codec_group_name(codec: str):
    if codec.startswith("g711"):
        return "g711"
    return codec


def enrich_with_clean_ratios(results):
    clean = next((item for item in results if item["condition"] == "clean"), None)
    if clean is None:
        return results

    clean_eer = clean["eer_percent"]
    clean_min_dcf = clean["min_dcf"]

    for item in results:
        item["codec_group"] = codec_group_name(item["codec"])
        if clean_eer > 0:
            item["eer_ratio_vs_clean"] = item["eer_percent"] / clean_eer
        else:
            item["eer_ratio_vs_clean"] = None
        if clean_min_dcf > 0:
            item["min_dcf_ratio_vs_clean"] = item["min_dcf"] / clean_min_dcf
        else:
            item["min_dcf_ratio_vs_clean"] = None
    return results


def write_reports(results, csv_path: Path, json_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "condition",
            "codec",
            "codec_group",
            "bitrate",
            "num_trials",
            "num_utts",
            "eer_percent",
            "min_dcf",
            "eer_ratio_vs_clean",
            "min_dcf_ratio_vs_clean",
        ])
        for item in results:
            writer.writerow(
                [
                    item["condition"],
                    item["codec"],
                    item.get("codec_group", codec_group_name(item["codec"])),
                    item["bitrate"],
                    item["num_trials"],
                    item["num_utts"],
                    f"{item['eer_percent']:.4f}",
                    f"{item['min_dcf']:.6f}",
                    "" if item.get("eer_ratio_vs_clean") is None else f"{item['eer_ratio_vs_clean']:.4f}",
                    "" if item.get("min_dcf_ratio_vs_clean") is None else f"{item['min_dcf_ratio_vs_clean']:.4f}",
                ]
            )

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def build_grouped_ranking_rows(results):
    grouped_rows = []
    candidates = [item for item in results if item["condition"] != "clean"]
    groups = sorted({item.get("codec_group", codec_group_name(item["codec"])) for item in candidates})

    for group in groups:
        group_items = [
            item for item in candidates if item.get("codec_group", codec_group_name(item["codec"])) == group
        ]
        group_items.sort(key=lambda x: (x["eer_percent"], x["min_dcf"]))
        for rank, item in enumerate(group_items, start=1):
            grouped_rows.append(
                {
                    "codec_group": group,
                    "rank_in_codec": rank,
                    "condition": item["condition"],
                    "codec": item["codec"],
                    "bitrate": item["bitrate"],
                    "eer_percent": item["eer_percent"],
                    "min_dcf": item["min_dcf"],
                    "eer_ratio_vs_clean": item.get("eer_ratio_vs_clean"),
                    "min_dcf_ratio_vs_clean": item.get("min_dcf_ratio_vs_clean"),
                }
            )
    return grouped_rows


def write_grouped_ranking_csv(rows, ranking_csv: Path):
    ranking_csv.parent.mkdir(parents=True, exist_ok=True)
    with ranking_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "codec_group",
                "rank_in_codec",
                "condition",
                "codec",
                "bitrate",
                "eer_percent",
                "min_dcf",
                "eer_ratio_vs_clean",
                "min_dcf_ratio_vs_clean",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["codec_group"],
                    row["rank_in_codec"],
                    row["condition"],
                    row["codec"],
                    row["bitrate"],
                    f"{row['eer_percent']:.4f}",
                    f"{row['min_dcf']:.6f}",
                    "" if row["eer_ratio_vs_clean"] is None else f"{row['eer_ratio_vs_clean']:.4f}",
                    ""
                    if row["min_dcf_ratio_vs_clean"] is None
                    else f"{row['min_dcf_ratio_vs_clean']:.4f}",
                ]
            )


def write_grouped_ranking_md(rows, ranking_md: Path):
    ranking_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| codec_group | rank_in_codec | condition | codec | bitrate | EER(%) | minDCF | EER/Clean | minDCF/Clean |",
        "|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {codec_group} | {rank_in_codec} | {condition} | {codec} | {bitrate} | {eer:.4f} | {min_dcf:.6f} | {eer_ratio} | {dcf_ratio} |".format(
                codec_group=row["codec_group"],
                rank_in_codec=row["rank_in_codec"],
                condition=row["condition"],
                codec=row["codec"],
                bitrate=row["bitrate"],
                eer=row["eer_percent"],
                min_dcf=row["min_dcf"],
                eer_ratio="" if row["eer_ratio_vs_clean"] is None else f"{row['eer_ratio_vs_clean']:.4f}",
                dcf_ratio=""
                if row["min_dcf_ratio_vs_clean"] is None
                else f"{row['min_dcf_ratio_vs_clean']:.4f}",
            )
        )
    ranking_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    check_ffmpeg()

    test_wav_scp = Path(args.test_wav_scp).resolve()#包含测试集中所有人的单条enroll语音与所有test语音的路径
    trials_file = Path(args.trials_file).resolve()#包含测试集中所有人enroll语音的路径
    model_bin = Path(args.model_bin).resolve()
    degraded_root = Path(args.degraded_root).resolve()
    embedding_cache_root = Path(args.embedding_cache_root).resolve()
    report_csv = Path(args.report_csv).resolve()
    report_json = Path(args.report_json).resolve()
    ranking_csv = Path(args.ranking_csv).resolve() if args.ranking_csv else None
    ranking_md = Path(args.ranking_md).resolve() if args.ranking_md else None
    overwrite_embeddings = args.overwrite or args.overwrite_embeddings

    if not test_wav_scp.exists():
        raise FileNotFoundError(f"test wav.scp not found: {test_wav_scp}")
    if not trials_file.exists():
        raise FileNotFoundError(f"trials file not found: {trials_file}")
    if not model_bin.exists():
        raise FileNotFoundError(f"model bin not found: {model_bin}")

    # Keep backward-compatible CLI while making it explicit that audio is no
    # longer persisted as degraded wav sets.
    if args.degraded_root:
        print(f"[INFO] --degraded_root is ignored in embedding-cache mode: {degraded_root}")

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
    )#从测试集中去一部分人的语音用于测试，进行enroll
    print(
        f"Trial selection: mode={args.trial_sample_mode}, stratified={args.stratified_sampling}, fraction={args.fraction}, "
        f"limit={args.limit}, seed={args.trial_sample_seed}, selected={len(trial_lines)}"
    )
    parsed_trials = parse_trials_records(trial_lines)
    trial_utts = extract_utts_from_trials(parsed_trials)

    missing = [utt for utt in trial_utts if utt not in wav_map]
    if missing:
        example = missing[:10]
        raise RuntimeError(
            f"{len(missing)} utterances in trials are missing in wav.scp after key normalization, first few: {example}"
        )

    clean_trial_map = {utt: wav_map[utt] for utt in trial_utts}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = init_campp_model(model_bin, device)
    feature_extractor = FBank(80, sample_rate=args.sample_rate, mean_nor=True)

    results = []

    print("Evaluating clean baseline...")
    clean_embeddings = compute_cached_embeddings(
        utt_to_clean_wav=clean_trial_map,
        condition_tag="clean",
        cache_root=embedding_cache_root,
        model=model,
        feature_extractor=feature_extractor,
        device=device,
        sample_rate=args.sample_rate,
        overwrite_embeddings=overwrite_embeddings,
        codec=None,
        bitrate=None,
    )
    clean_labels, clean_scores = score_trials(parsed_trials, clean_embeddings)
    clean_eer, clean_min_dcf = evaluate_metrics(clean_labels, clean_scores, args.p_target, args.c_miss, args.c_fa)
    results.append(
        {
            "condition": "clean",
            "codec": "clean",
            "bitrate": "-",
            "num_trials": len(parsed_trials),
            "num_utts": len(trial_utts),
            "eer_percent": clean_eer,
            "min_dcf": clean_min_dcf,
        }
    )
    print(f"[clean] EER={clean_eer:.4f}% minDCF={clean_min_dcf:.6f}")

    conditions = []
    for b in parse_csv_list(args.opus_bitrates):
        conditions.append(("opus", b))
    for b in parse_csv_list(args.amrwb_bitrates):
        conditions.append(("amrwb", b))
    for b in parse_csv_list(args.aac_bitrates):
        conditions.append(("aac", b))
    for variant in parse_csv_list(args.g711_variants):
        conditions.append((variant, "64k"))

    for codec, bitrate in conditions:
        print(f"Running condition codec={codec}, bitrate={bitrate}")
        condition_tag = f"{codec}_{bitrate.replace('.', '').replace('/', '_')}"
        cond_embeddings = compute_cached_embeddings(
            utt_to_clean_wav=clean_trial_map,
            condition_tag=condition_tag,
            cache_root=embedding_cache_root,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            sample_rate=args.sample_rate,
            overwrite_embeddings=overwrite_embeddings,
            codec=codec,
            bitrate=bitrate,
        )
        labels, scores = score_trials(parsed_trials, cond_embeddings)
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

    enrich_with_clean_ratios(results)
    write_reports(results, report_csv, report_json)

    if ranking_csv or ranking_md:
        grouped_rows = build_grouped_ranking_rows(results)
        if ranking_csv:
            write_grouped_ranking_csv(grouped_rows, ranking_csv)
            print(f"Saved grouped ranking CSV: {ranking_csv}")
        if ranking_md:
            write_grouped_ranking_md(grouped_rows, ranking_md)
            print(f"Saved grouped ranking Markdown: {ranking_md}")

    print(f"Saved CSV report: {report_csv}")
    print(f"Saved JSON report: {report_json}")


if __name__ == "__main__":
    main()
