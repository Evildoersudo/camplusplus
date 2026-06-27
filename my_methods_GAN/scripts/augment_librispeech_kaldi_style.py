#!/usr/bin/env python3
"""Create Kaldi-style acoustic augmentation for a LibriSpeech wav tree.

The output keeps the same speaker/chapter/utterance layout as the input. Each
input utterance produces one target wav: clean, reverb, noise, music, or babble.
Codec conversion should be run after this step so the training pair is
input=codec(x_aug), target=x_aug.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import math
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
DEFAULT_LIBRISPEECH_ROOT = Path("my_methods_GAN/data/LibriSpeech/train-clean-100")
DEFAULT_OUTPUT_ROOT = Path("my_methods_GAN/data/LibriSpeech/train-clean-100-kaldi-aug")
DEFAULT_RIR_ROOT = Path("egs/3dspeaker/sv-cam++/data/raw_data/RIRS_NOISES")
DEFAULT_MUSAN_ROOT = Path("egs/3dspeaker/sv-cam++/data/raw_data/musan")


@dataclass(frozen=True)
class AugmentTask:
    source: Path
    relative: Path
    output: Path
    aug_type: str


@dataclass(frozen=True)
class TaskResult:
    rel_path: str
    utt_id: str
    spk_id: str
    output_wav: str
    aug_type: str
    snr_db: str
    num_noise_sources: str
    noise_paths: str
    rir_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LibriSpeech-layout augmented wav tree with clean/reverb/"
            "noise/music/babble proportions."
        )
    )
    parser.add_argument("--input_root", type=Path, default=DEFAULT_LIBRISPEECH_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rir_root", type=Path, default=DEFAULT_RIR_ROOT)
    parser.add_argument("--musan_root", type=Path, default=DEFAULT_MUSAN_ROOT)
    parser.add_argument(
        "--include_manifest",
        "--include-manifest",
        type=Path,
        default=None,
        help="Optional clean manifest; rows may contain clean_wav or rel_path.",
    )
    parser.add_argument(
        "--metadata_csv",
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional CSV recording the augmentation choice for each utterance.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument(
        "--clean_ratio",
        type=float,
        default=0.5,
        help="Fraction left without acoustic augmentation.",
    )
    parser.add_argument("--reverb_ratio", type=float, default=0.125)
    parser.add_argument("--noise_ratio", type=float, default=0.125)
    parser.add_argument("--music_ratio", type=float, default=0.125)
    parser.add_argument("--babble_ratio", type=float, default=0.125)
    parser.add_argument(
        "--noise_snrs",
        default="0,5,10,15",
        help="Comma-separated SNR choices for MUSAN noise.",
    )
    parser.add_argument(
        "--music_snrs",
        default="5,8,10,15",
        help="Comma-separated SNR choices for MUSAN music.",
    )
    parser.add_argument(
        "--babble_snrs",
        default="13,15,17,20",
        help="Comma-separated SNR choices for MUSAN speech babble.",
    )
    parser.add_argument("--babble_min_sources", type=int, default=3)
    parser.add_argument("--babble_max_sources", type=int, default=7)
    parser.add_argument(
        "--peak_limit",
        type=float,
        default=0.95,
        help="Final ffmpeg alimiter peak limit.",
    )
    return parser.parse_args()


def run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        tail = "\n".join(detail[-12:])
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(command)}\n{tail}") from exc


def ffprobe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return max(0.001, float(result.stdout.strip()))


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty float list: {raw!r}")
    return values


def collect_audio(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    )


def collect_librispeech_wavs(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/*.wav"))


def relative_manifest_path(path_text: str, roots: list[Path]) -> str:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(path.resolve())
    for candidate in candidates:
        for root in roots:
            try:
                return candidate.relative_to(root).as_posix()
            except ValueError:
                pass
    if not path.is_absolute() and len(path.parts) < 3:
        return path.as_posix()
    absolute = path if path.is_absolute() else path.resolve()
    for root in roots:
        try:
            return absolute.relative_to(root).as_posix()
        except ValueError:
            pass
    if len(absolute.parts) >= 3:
        return Path(*absolute.parts[-3:]).as_posix()
    raise ValueError(f"Cannot infer speaker/chapter/file relative path: {path_text}")


def load_include_relpaths(manifest_path: Path, roots: list[Path]) -> set[str]:
    relpaths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        for row_number, row in enumerate(reader, 2):
            path_text = row.get("rel_path") or row.get("clean_wav")
            if not path_text:
                raise ValueError(
                    f"Manifest row {row_number} must contain rel_path or clean_wav"
                )
            relpaths.add(relative_manifest_path(path_text, roots))
    if not relpaths:
        raise ValueError(f"No rows found in include manifest: {manifest_path}")
    return relpaths


def stable_rng(seed: int, rel_path: Path, aug_type: str) -> random.Random:
    key = f"{seed}:{rel_path.as_posix()}:{aug_type}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return random.Random(int(digest[:16], 16))


def assign_augment_types(
    wavs: list[Path],
    input_root: Path,
    output_root: Path,
    ratios: dict[str, float],
    seed: int,
) -> list[AugmentTask]:
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("At least one augmentation ratio must be positive.")

    normalized = {key: value / total_ratio for key, value in ratios.items()}
    ordered_types = ["clean", "reverb", "noise", "music", "babble"]
    counts: dict[str, int] = {}
    assigned = 0
    for aug_type in ordered_types[:-1]:
        counts[aug_type] = int(round(len(wavs) * normalized[aug_type]))
        assigned += counts[aug_type]
    counts[ordered_types[-1]] = len(wavs) - assigned

    labels: list[str] = []
    for aug_type in ordered_types:
        labels.extend([aug_type] * max(0, counts[aug_type]))
    labels = labels[: len(wavs)]
    while len(labels) < len(wavs):
        labels.append("clean")

    rng = random.Random(seed)
    shuffled = list(wavs)
    rng.shuffle(shuffled)

    label_by_path = dict(zip(shuffled, labels, strict=True))
    tasks = []
    for wav in wavs:
        relative = wav.relative_to(input_root)
        tasks.append(
            AugmentTask(
                source=wav,
                relative=relative,
                output=output_root / relative,
                aug_type=label_by_path[wav],
            )
        )
    return tasks


def convert_to_wav16(source: Path, destination: Path, sample_rate: int) -> None:
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def copy_clean(source: Path, destination: Path, sample_rate: int, peak_limit: float) -> None:
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-af",
            f"alimiter=limit={peak_limit}",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def wav_rms(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        frames = handle.readframes(handle.getnframes())
    if not frames:
        return 1.0
    total = 0
    count = 0
    for index in range(0, len(frames), 2):
        sample = int.from_bytes(frames[index : index + 2], "little", signed=True)
        total += sample * sample
        count += 1
    if count == 0:
        return 1.0
    return max(1.0, math.sqrt(total / count))


def normalize_rir(source: Path, destination: Path, sample_rate: int) -> None:
    with tempfile.TemporaryDirectory(prefix="rir16_") as tmp:
        tmp_wav = Path(tmp) / "rir.wav"
        convert_to_wav16(source, tmp_wav, sample_rate)
        with wave.open(str(tmp_wav), "rb") as handle:
            params = handle.getparams()
            frames = handle.readframes(handle.getnframes())

        samples = [
            int.from_bytes(frames[i : i + 2], "little", signed=True)
            for i in range(0, len(frames), 2)
        ]
        energy = math.sqrt(sum((sample / 32768.0) ** 2 for sample in samples))
        if energy <= 1e-12:
            shutil.copyfile(tmp_wav, destination)
            return
        scaled = bytearray()
        for sample in samples:
            value = int(round((sample / 32768.0) / energy * 32767.0))
            value = max(-32768, min(32767, value))
            scaled.extend(value.to_bytes(2, "little", signed=True))

        with wave.open(str(destination), "wb") as handle:
            handle.setparams(params)
            handle.writeframes(bytes(scaled))


def make_noise_segment(
    source: Path,
    destination: Path,
    duration: float,
    sample_rate: int,
    rng: random.Random,
) -> None:
    source_duration = ffprobe_duration(source)
    offset = 0.0
    if source_duration > duration + 0.25:
        offset = rng.uniform(0.0, source_duration - duration)
    command = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(source),
    ]
    if offset > 0:
        command.extend(["-ss", f"{offset:.3f}"])
    command.extend(
        [
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    run_ffmpeg(command)


def mix_with_noise(
    speech_wav: Path,
    noise_wav: Path,
    output_wav: Path,
    sample_rate: int,
    snr_db: float,
    peak_limit: float,
) -> None:
    speech_rms = wav_rms(speech_wav)
    noise_rms = wav_rms(noise_wav)
    scale = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(speech_wav),
            "-i",
            str(noise_wav),
            "-filter_complex",
            (
                f"[1:a]volume={scale:.8f}[n];"
                f"[0:a][n]amix=inputs=2:duration=first:dropout_transition=0,"
                f"alimiter=limit={peak_limit}[out]"
            ),
            "-map",
            "[out]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )


def apply_reverb(
    speech_wav: Path,
    rir_source: Path,
    output_wav: Path,
    duration: float,
    sample_rate: int,
    peak_limit: float,
    tmp_dir: Path,
) -> Path:
    rir_wav = tmp_dir / "rir_norm.wav"
    normalize_rir(rir_source, rir_wav, sample_rate)
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(speech_wav),
            "-i",
            str(rir_wav),
            "-filter_complex",
            (
                "[0:a]aformat=sample_fmts=fltp:channel_layouts=mono[sp];"
                "[1:a]aformat=sample_fmts=fltp:channel_layouts=mono[rir];"
                f"[sp][rir]afir=dry=0:wet=1,atrim=0:{duration:.3f},"
                f"asetpts=N/SR/TB,alimiter=limit={peak_limit}[out]"
            ),
            "-map",
            "[out]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )
    return rir_source


def make_babble(
    speech_sources: list[Path],
    destination: Path,
    duration: float,
    sample_rate: int,
    rng: random.Random,
    tmp_dir: Path,
) -> list[Path]:
    segment_paths = []
    for index, source in enumerate(speech_sources):
        segment = tmp_dir / f"babble_src_{index}.wav"
        make_noise_segment(source, segment, duration, sample_rate, rng)
        segment_paths.append(segment)

    command = ["ffmpeg", "-y"]
    for segment in segment_paths:
        command.extend(["-i", str(segment)])
    inputs = "".join(f"[{i}:a]" for i in range(len(segment_paths)))
    command.extend(
        [
            "-filter_complex",
            f"{inputs}amix=inputs={len(segment_paths)}:duration=first:"
            "dropout_transition=0,volume=1[out]",
            "-map",
            "[out]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    run_ffmpeg(command)
    return speech_sources


def process_task(
    task: AugmentTask,
    rir_small: list[Path],
    rir_medium: list[Path],
    musan_noise: list[Path],
    musan_music: list[Path],
    musan_speech: list[Path],
    noise_snrs: list[float],
    music_snrs: list[float],
    babble_snrs: list[float],
    babble_min_sources: int,
    babble_max_sources: int,
    sample_rate: int,
    seed: int,
    peak_limit: float,
    overwrite: bool,
) -> TaskResult:
    if task.output.exists() and task.output.stat().st_size > 44 and not overwrite:
        return task_result(task, "", "", [], "")

    task.output.parent.mkdir(parents=True, exist_ok=True)
    rng = stable_rng(seed, task.relative, task.aug_type)
    snr = ""
    noise_paths: list[Path] = []
    rir_path = ""
    num_noise_sources = ""

    with tempfile.TemporaryDirectory(prefix="kaldi_aug_") as tmp:
        tmp_dir = Path(tmp)
        speech_wav = tmp_dir / "speech.wav"
        convert_to_wav16(task.source, speech_wav, sample_rate)
        duration = ffprobe_duration(speech_wav)

        if task.aug_type == "clean":
            copy_clean(speech_wav, task.output, sample_rate, peak_limit)
        elif task.aug_type == "reverb":
            room_rirs = rir_small if rng.random() < 0.5 else rir_medium
            rir_source = rng.choice(room_rirs)
            rir_path = str(rir_source)
            apply_reverb(
                speech_wav,
                rir_source,
                task.output,
                duration,
                sample_rate,
                peak_limit,
                tmp_dir,
            )
        elif task.aug_type in {"noise", "music"}:
            source_pool = musan_noise if task.aug_type == "noise" else musan_music
            snr_choices = noise_snrs if task.aug_type == "noise" else music_snrs
            snr_value = rng.choice(snr_choices)
            noise_source = rng.choice(source_pool)
            noise_wav = tmp_dir / f"{task.aug_type}.wav"
            make_noise_segment(noise_source, noise_wav, duration, sample_rate, rng)
            mix_with_noise(
                speech_wav,
                noise_wav,
                task.output,
                sample_rate,
                snr_value,
                peak_limit,
            )
            snr = f"{snr_value:g}"
            noise_paths = [noise_source]
            num_noise_sources = "1"
        elif task.aug_type == "babble":
            snr_value = rng.choice(babble_snrs)
            num_sources = rng.randint(babble_min_sources, babble_max_sources)
            speech_sources = [rng.choice(musan_speech) for _ in range(num_sources)]
            babble_wav = tmp_dir / "babble.wav"
            noise_paths = make_babble(
                speech_sources,
                babble_wav,
                duration,
                sample_rate,
                rng,
                tmp_dir,
            )
            mix_with_noise(
                speech_wav,
                babble_wav,
                task.output,
                sample_rate,
                snr_value,
                peak_limit,
            )
            snr = f"{snr_value:g}"
            num_noise_sources = str(num_sources)
        else:
            raise ValueError(f"Unsupported augmentation type: {task.aug_type}")

    return task_result(task, snr, num_noise_sources, noise_paths, rir_path)


def task_result(
    task: AugmentTask,
    snr: str,
    num_noise_sources: str,
    noise_paths: list[Path],
    rir_path: str,
) -> TaskResult:
    rel = task.relative.as_posix()
    utt_id = task.relative.with_suffix("").as_posix().replace("/", "-")
    spk_id = task.relative.parts[0]
    return TaskResult(
        rel_path=rel,
        utt_id=utt_id,
        spk_id=spk_id,
        output_wav=str(task.output),
        aug_type=task.aug_type,
        snr_db=snr,
        num_noise_sources=num_noise_sources,
        noise_paths=";".join(str(path) for path in noise_paths),
        rir_path=rir_path,
    )


def validate_assets(
    rir_root: Path,
    musan_root: Path,
) -> tuple[list[Path], list[Path], list[Path], list[Path], list[Path]]:
    simulated = rir_root / "simulated_rirs"
    rir_small = collect_audio(simulated / "smallroom")
    rir_medium = collect_audio(simulated / "mediumroom")
    musan_noise = collect_audio(musan_root / "noise")
    musan_music = collect_audio(musan_root / "music")
    musan_speech = collect_audio(musan_root / "speech")

    missing = []
    for name, paths in [
        ("RIR smallroom", rir_small),
        ("RIR mediumroom", rir_medium),
        ("MUSAN noise", musan_noise),
        ("MUSAN music", musan_music),
        ("MUSAN speech", musan_speech),
    ]:
        if not paths:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing augmentation assets: {', '.join(missing)}")
    return rir_small, rir_medium, musan_noise, musan_music, musan_speech


def write_metadata(metadata_csv: Path, results: list[TaskResult]) -> None:
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "utt_id",
                "spk_id",
                "rel_path",
                "clean_wav",
                "aug_type",
                "snr_db",
                "num_noise_sources",
                "noise_paths",
                "rir_path",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.utt_id,
                    row.spk_id,
                    row.rel_path,
                    row.output_wav,
                    row.aug_type,
                    row.snr_db,
                    row.num_noise_sources,
                    row.noise_paths,
                    row.rir_path,
                ]
            )


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    rir_root = args.rir_root.resolve()
    musan_root = args.musan_root.resolve()
    include_manifest = args.include_manifest.resolve() if args.include_manifest else None
    metadata_csv = (
        args.metadata_csv.resolve()
        if args.metadata_csv
        else output_root.parent / f"{output_root.name}_augmentation_manifest.csv"
    )

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.babble_min_sources < 1 or args.babble_max_sources < args.babble_min_sources:
        raise ValueError("Invalid babble source range.")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing input root: {input_root}")
    if include_manifest is not None and not include_manifest.is_file():
        raise FileNotFoundError(f"Missing include manifest: {include_manifest}")

    wavs = collect_librispeech_wavs(input_root)
    if include_manifest is not None:
        include_relpaths = load_include_relpaths(include_manifest, [input_root])
        wavs = [
            path
            for path in wavs
            if path.relative_to(input_root).as_posix() in include_relpaths
        ]
    if not wavs:
        raise RuntimeError(f"No LibriSpeech-layout wav files found under {input_root}")

    rir_small, rir_medium, musan_noise, musan_music, musan_speech = validate_assets(
        rir_root, musan_root
    )
    noise_snrs = parse_float_list(args.noise_snrs)
    music_snrs = parse_float_list(args.music_snrs)
    babble_snrs = parse_float_list(args.babble_snrs)

    ratios = {
        "clean": args.clean_ratio,
        "reverb": args.reverb_ratio,
        "noise": args.noise_ratio,
        "music": args.music_ratio,
        "babble": args.babble_ratio,
    }
    tasks = assign_augment_types(wavs, input_root, output_root, ratios, args.seed)
    counts = {key: 0 for key in ratios}
    for task in tasks:
        counts[task.aug_type] += 1

    print(f"Input root:      {input_root}")
    print(f"Output root:     {output_root}")
    print(f"Metadata CSV:    {metadata_csv}")
    print(f"Utterances:      {len(tasks)}")
    print(
        "Aug counts:      "
        + " ".join(f"{key}={counts[key]}" for key in ["clean", "reverb", "noise", "music", "babble"])
    )
    print(f"RIR small/medium:{len(rir_small)}/{len(rir_medium)}")
    print(
        f"MUSAN n/m/s:     {len(musan_noise)}/{len(musan_music)}/{len(musan_speech)}"
    )

    if args.dry_run:
        print("Dry run complete; no files were written.")
        return 0

    results: list[TaskResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_task,
                task,
                rir_small,
                rir_medium,
                musan_noise,
                musan_music,
                musan_speech,
                noise_snrs,
                music_snrs,
                babble_snrs,
                args.babble_min_sources,
                args.babble_max_sources,
                args.sample_rate,
                args.seed,
                args.peak_limit,
                args.overwrite,
            ): task
            for task in tasks
        }
        total = len(futures)
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"Augmentation failed for {task.source}") from exc
            if completed % 500 == 0 or completed == total:
                print(f"Processed:       {completed}/{total}")

    results.sort(key=lambda row: row.rel_path)
    write_metadata(metadata_csv, results)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
