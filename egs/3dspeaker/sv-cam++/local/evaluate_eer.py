import argparse
from pathlib import Path

from modelscope.pipelines import pipeline
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def parse_args():
    # The script accepts the current VCTK-style trials where each line is
    # "utt1 utt2 target" or "utt1 utt2 nontarget".
    parser = argparse.ArgumentParser(description="Evaluate speaker verification EER with a ModelScope pretrained model.")
    parser.add_argument("--model_id", type=str, default="iic/speech_campplus_sv_zh-cn_16k-common", help="ModelScope model id")
    parser.add_argument("--trials_file", type=str, required=True, help="Trials file")
    parser.add_argument("--wav_root", type=str, default="", help="Root directory containing wavs for single-set evaluation")
    parser.add_argument("--clean_wav_root", type=str, default="", help="Root directory of clean wavs")
    parser.add_argument("--degraded_wav_root", type=str, default="", help="Root directory of degraded wavs")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N trials, 0 means all")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Evaluate only the first fraction of trials, e.g. 0.1 or 0.2",
    )
    return parser.parse_args()


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100.0


def resolve_wav_path(wav_root: Path, utt_id: str) -> Path:
    # VCTK utterance ids are formatted as p225_001, so the speaker id is the
    # prefix before the first underscore.
    speaker_id = utt_id.split("_", 1)[0]
    wav_path = wav_root / speaker_id / f"{utt_id}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"Wav file not found for utterance {utt_id}: {wav_path}")
    return wav_path


def load_trials(trials_file: Path, limit: int, fraction: float):
    lines = [line.strip() for line in trials_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 < fraction <= 1.0:
        raise ValueError("--fraction must be in the range (0, 1].")
    if fraction < 1.0:
        keep = max(1, int(len(lines) * fraction))
        lines = lines[:keep]
    if limit > 0:
        lines = lines[:limit]
    return lines


def evaluate_trials(sv_pipeline, trials, wav_root: Path, tag: str):
    labels = []
    scores = []
    print(f"Evaluating {tag} on {len(trials)} trial pairs")

    for idx, line in enumerate(trials, start=1):
        utt1, utt2, label = line.split()
        wav1 = resolve_wav_path(wav_root, utt1)
        wav2 = resolve_wav_path(wav_root, utt2)

        result = sv_pipeline([str(wav1), str(wav2)])
        score = result["score"]
        labels.append(1 if label in ("1", "target") else 0)
        scores.append(score)

        if idx % 100 == 0 or idx == len(trials):
            print(f"[{tag}] Processed {idx}/{len(trials)} pairs")

    eer = compute_eer(labels, scores)
    print(f"{tag} EER: {eer:.3f}%")
    return eer


def main():
    args = parse_args()
    trials_file = Path(args.trials_file).resolve()

    if not trials_file.exists():
        raise FileNotFoundError(f"Trials file not found: {trials_file}")

    print("Loading pretrained speaker verification pipeline...")
    sv_pipeline = pipeline(task="speaker-verification", model=args.model_id)

    trials = load_trials(trials_file, args.limit, args.fraction)

    if args.wav_root:
        wav_root = Path(args.wav_root).resolve()
        if not wav_root.exists():
            raise FileNotFoundError(f"Wav root not found: {wav_root}")
        evaluate_trials(sv_pipeline, trials, wav_root, tag="Single-set")
        return

    if not args.clean_wav_root or not args.degraded_wav_root:
        raise ValueError("Provide either --wav_root, or both --clean_wav_root and --degraded_wav_root.")

    clean_wav_root = Path(args.clean_wav_root).resolve()
    degraded_wav_root = Path(args.degraded_wav_root).resolve()
    if not clean_wav_root.exists():
        raise FileNotFoundError(f"Clean wav root not found: {clean_wav_root}")
    if not degraded_wav_root.exists():
        raise FileNotFoundError(f"Degraded wav root not found: {degraded_wav_root}")

    clean_eer = evaluate_trials(sv_pipeline, trials, clean_wav_root, tag="Clean")
    degraded_eer = evaluate_trials(sv_pipeline, trials, degraded_wav_root, tag="Degraded")

    print("Summary")
    print(f"Clean EER: {clean_eer:.3f}%")
    print(f"Degraded EER: {degraded_eer:.3f}%")
    if clean_eer > 0:
        print(f"EER degradation factor: {degraded_eer / clean_eer:.3f}x")
    else:
        print("EER degradation factor: undefined because clean EER is 0")


if __name__ == "__main__":
    main()
