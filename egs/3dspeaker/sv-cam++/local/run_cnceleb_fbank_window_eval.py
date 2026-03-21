import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as Kaldi

try:
    from speakerlab.utils.builder import dynamic_import
    from speakerlab.utils.score_metrics import compute_c_norm, compute_eer, compute_pmiss_pfa_rbst
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4]))
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
        description="Evaluate CAM++ under different FBank window lengths without modifying existing scripts."
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
        default="egs/3dspeaker/sv-cam++/exp/fbank_window_eval/embedding_cache",
        help="Root directory to store cached embedding .npy files",
    )
    parser.add_argument(
        "--report_csv",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/fbank_window_eval/window_eval_report.csv",
        help="Summary report in CSV",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/fbank_window_eval/window_eval_report.json",
        help="Summary report in JSON",
    )
    parser.add_argument(
        "--table_md",
        type=str,
        default="egs/3dspeaker/sv-cam++/exp/fbank_window_eval/window_eval_table.md",
        help="Markdown table for paper",
    )

    parser.add_argument(
        "--codec_conditions",
        type=str,
        default="clean,opus@4k,opus@8k,amrwb@8.85k",
        help="Comma-separated conditions. Format: clean or codec@bitrate",
    )
    parser.add_argument(
        "--window_lengths_ms",
        type=str,
        default="25,20,15,10",
        help="Comma-separated FBank window lengths in milliseconds",
    )
    parser.add_argument("--fbank_shift_ms", type=float, default=10.0, help="FBank frame shift in milliseconds")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of mel bins")

    parser.add_argument("--sample_rate", type=int, default=16000, help="Decoded wav sample rate")
    parser.add_argument("--fraction", type=float, default=1.0, help="Use fraction of trials before limit")
    parser.add_argument("--limit", type=int, default=0, help="Use first N (head mode) or sampled N (random mode) trials")
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

    parser.add_argument("--p_target", type=float, default=0.01, help="p_target in minDCF")
    parser.add_argument("--c_miss", type=float, default=1.0, help="c_miss in minDCF")
    parser.add_argument("--c_fa", type=float, default=1.0, help="c_fa in minDCF")
    return parser.parse_args()


def parse_csv_list(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_list(text: str):
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("--window_lengths_ms is empty")
    for v in values:
        if v <= 0:
            raise ValueError("window length must be > 0")
    return values


def format_ms_tag(ms: float):
    return str(ms).replace(".", "p")


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
        utt1 = normalize_trial_utt_key(utt1_raw)
        utt2 = normalize_trial_utt_key(utt2_raw)
        records.append((utt1, utt2, label))
    return records


def extract_utts_from_trials(parsed_trials):
    utts = set()
    for utt1, utt2, _ in parsed_trials:
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


def load_wav_mono_16k(wav_file: Path, target_sr: int):
    wav, sr = torchaudio.load(str(wav_file))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav[:1]
    return wav


def degrade_to_waveform(clean_wav: Path, codec: str, bitrate: str, sample_rate: int):
    with tempfile.TemporaryDirectory(prefix="fbank_window_tmp_") as td:
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


def compute_fbank(wav: torch.Tensor, n_mels: int, sample_rate: int, frame_length_ms: float, frame_shift_ms: float):
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav[:1]
    feat = Kaldi.fbank(
        wav,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        dither=0,
    )
    feat = feat - feat.mean(0, keepdim=True)
    return feat


def init_campp_model(model_bin: Path, device: torch.device):
    model = dynamic_import(CAMPPLUS_COMMON["obj"])(**CAMPPLUS_COMMON["args"])
    state = torch.load(str(model_bin), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_embedding_from_wav(wav, model, device, n_mels, sample_rate, frame_length_ms, frame_shift_ms):
    feat = compute_fbank(wav, n_mels, sample_rate, frame_length_ms, frame_shift_ms)
    feat = feat.unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(feat).detach().squeeze(0).cpu().numpy()
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding is encountered.")
    return vec / norm


def embedding_cache_path(cache_root: Path, window_tag: str, condition_tag: str, utt: str):
    safe = utt.replace("/", "_").replace("\\", "_")
    return cache_root / window_tag / condition_tag / f"{safe}.npy"


def compute_cached_embeddings(
    utt_to_clean_wav,
    window_tag: str,
    condition_tag: str,
    cache_root: Path,
    model,
    device: torch.device,
    sample_rate: int,
    n_mels: int,
    frame_length_ms: float,
    frame_shift_ms: float,
    overwrite_embeddings: bool,
    codec: str = None,
    bitrate: str = None,
):
    emb = {}
    generated = 0
    reused = 0

    utts = sorted(utt_to_clean_wav)
    for idx, utt in enumerate(utts, start=1):
        cache_file = embedding_cache_path(cache_root, window_tag, condition_tag, utt)
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
            vec = compute_embedding_from_wav(
                wav=wav,
                model=model,
                device=device,
                n_mels=n_mels,
                sample_rate=sample_rate,
                frame_length_ms=frame_length_ms,
                frame_shift_ms=frame_shift_ms,
            )
            np.save(cache_file, vec)
            emb[utt] = vec
            generated += 1

        if idx % 500 == 0 or idx == len(utts):
            print(
                f"[{window_tag}][{condition_tag}] progress: {idx}/{len(utts)} "
                f"(generated={generated}, reused={reused})"
            )

    return emb


def score_trials(parsed_trials, embeddings):
    labels = []
    scores = []
    for utt1, utt2, label in parsed_trials:
        v1 = embeddings[utt1]
        v2 = embeddings[utt2]
        scores.append(float(np.dot(v1, v2)))
        labels.append(1 if label in ("1", "target") else 0)
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


def write_reports(results, report_csv: Path, report_json: Path):
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    with report_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "window_ms",
                "shift_ms",
                "condition",
                "codec",
                "bitrate",
                "num_trials",
                "num_utts",
                "eer_percent",
                "min_dcf",
                "eer_ratio_vs_window25_clean",
                "min_dcf_ratio_vs_window25_clean",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    f"{r['window_ms']:.2f}",
                    f"{r['shift_ms']:.2f}",
                    r["condition"],
                    r["codec"],
                    r["bitrate"],
                    r["num_trials"],
                    r["num_utts"],
                    f"{r['eer_percent']:.4f}",
                    f"{r['min_dcf']:.6f}",
                    "" if r["eer_ratio_vs_window25_clean"] is None else f"{r['eer_ratio_vs_window25_clean']:.4f}",
                    "" if r["min_dcf_ratio_vs_window25_clean"] is None else f"{r['min_dcf_ratio_vs_window25_clean']:.4f}",
                ]
            )

    report_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


def write_markdown_table(results, table_md: Path, windows, condition_tags):
    table_md.parent.mkdir(parents=True, exist_ok=True)

    lines = ["## FBank窗长实验结果（EER% / minDCF）", ""]
    header = "| 条件 | " + " | ".join([f"w={w:.0f}ms" for w in windows]) + " |"
    sep = "|---|" + "---|" * len(windows)
    lines.append(header)
    lines.append(sep)

    value_map = {(r["condition"], r["window_ms"]): (r["eer_percent"], r["min_dcf"]) for r in results}
    for cond in condition_tags:
        vals = []
        for w in windows:
            key = (cond, w)
            if key in value_map:
                eer, dcf = value_map[key]
                vals.append(f"{eer:.3f} / {dcf:.4f}")
            else:
                vals.append("-")
        lines.append("| " + cond + " | " + " | ".join(vals) + " |")

    table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def enrich_ratios(results):
    baseline = next((r for r in results if abs(r["window_ms"] - 25.0) < 1e-9 and r["condition"] == "clean"), None)
    if baseline is None:
        return

    base_eer = baseline["eer_percent"]
    base_dcf = baseline["min_dcf"]
    for r in results:
        r["eer_ratio_vs_window25_clean"] = None if base_eer <= 0 else r["eer_percent"] / base_eer
        r["min_dcf_ratio_vs_window25_clean"] = None if base_dcf <= 0 else r["min_dcf"] / base_dcf


def main():
    args = parse_args()
    check_ffmpeg()

    if args.fbank_shift_ms <= 0:
        raise ValueError("--fbank_shift_ms must be > 0")
    if args.n_mels <= 0:
        raise ValueError("--n_mels must be > 0")

    test_wav_scp = Path(args.test_wav_scp).resolve()
    trials_file = Path(args.trials_file).resolve()
    model_bin = Path(args.model_bin).resolve()
    cache_root = Path(args.embedding_cache_root).resolve()
    report_csv = Path(args.report_csv).resolve()
    report_json = Path(args.report_json).resolve()
    table_md = Path(args.table_md).resolve() if args.table_md else None

    if not test_wav_scp.exists():
        raise FileNotFoundError(f"test wav.scp not found: {test_wav_scp}")
    if not trials_file.exists():
        raise FileNotFoundError(f"trials file not found: {trials_file}")
    if not model_bin.exists():
        raise FileNotFoundError(f"model bin not found: {model_bin}")

    windows = parse_float_list(args.window_lengths_ms)
    condition_specs = [parse_condition(x) for x in parse_csv_list(args.codec_conditions)]
    condition_tags = ["clean" if c == "clean" else f"{c}_{b.replace('.', '').replace('/', '_')}" for c, b in condition_specs]

    trial_lines = load_trials(
        trials_file=trials_file,
        fraction=args.fraction,
        limit=args.limit,
        sample_mode=args.trial_sample_mode,
        sample_seed=args.trial_sample_seed,
    )
    print(
        f"Trial selection: mode={args.trial_sample_mode}, fraction={args.fraction}, "
        f"limit={args.limit}, seed={args.trial_sample_seed}, selected={len(trial_lines)}"
    )

    parsed_trials = parse_trials_records(trial_lines)
    trial_utts = extract_utts_from_trials(parsed_trials)

    wav_map = load_wavscp(test_wav_scp)
    missing = [utt for utt in trial_utts if utt not in wav_map]
    if missing:
        raise RuntimeError(
            f"{len(missing)} utterances in trials are missing in wav.scp after key normalization, first few: {missing[:10]}"
        )
    clean_trial_map = {utt: wav_map[utt] for utt in trial_utts}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = init_campp_model(model_bin, device)

    results = []
    for w in windows:
        window_tag = f"fbank_w{format_ms_tag(w)}_h{format_ms_tag(args.fbank_shift_ms)}"
        print(f"Running window={w}ms, shift={args.fbank_shift_ms}ms, cache_namespace={window_tag}")

        for codec, bitrate in condition_specs:
            condition_tag = "clean" if codec == "clean" else f"{codec}_{bitrate.replace('.', '').replace('/', '_')}"
            emb = compute_cached_embeddings(
                utt_to_clean_wav=clean_trial_map,
                window_tag=window_tag,
                condition_tag=condition_tag,
                cache_root=cache_root,
                model=model,
                device=device,
                sample_rate=args.sample_rate,
                n_mels=args.n_mels,
                frame_length_ms=w,
                frame_shift_ms=args.fbank_shift_ms,
                overwrite_embeddings=args.overwrite_embeddings,
                codec=None if codec == "clean" else codec,
                bitrate=None if codec == "clean" else bitrate,
            )
            labels, scores = score_trials(parsed_trials, emb)
            eer, min_dcf = evaluate_metrics(labels, scores, args.p_target, args.c_miss, args.c_fa)
            results.append(
                {
                    "window_ms": float(w),
                    "shift_ms": float(args.fbank_shift_ms),
                    "condition": condition_tag,
                    "codec": codec,
                    "bitrate": bitrate,
                    "num_trials": len(parsed_trials),
                    "num_utts": len(trial_utts),
                    "eer_percent": eer,
                    "min_dcf": min_dcf,
                }
            )
            print(f"[{window_tag}][{condition_tag}] EER={eer:.4f}% minDCF={min_dcf:.6f}")

    enrich_ratios(results)
    write_reports(results, report_csv, report_json)
    print(f"Saved CSV report: {report_csv}")
    print(f"Saved JSON report: {report_json}")

    if table_md:
        write_markdown_table(results, table_md, windows, condition_tags)
        print(f"Saved Markdown table: {table_md}")


if __name__ == "__main__":
    main()
