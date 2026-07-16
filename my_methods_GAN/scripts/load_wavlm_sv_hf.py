#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/load microsoft/wavlm-base-plus-sv and run a small similarity demo.")
    p.add_argument("--model_id", type=str, default="microsoft/wavlm-base-plus-sv")
    p.add_argument("--output_dir", type=str, default="my_methods_GAN/pretrained/WavLM_SV")
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--threshold", type=float, default=0.86)
    p.add_argument("--skip_demo", action="store_true", help="Only download/save model files, skip dataset demo.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    except Exception as exc:
        raise RuntimeError(
            "Missing transformers dependency. Please install it first, e.g. pip install transformers"
        ) from exc

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    print(f"[info] model_id={args.model_id}")
    print(f"[info] output_dir={out_dir}")
    if cache_dir is not None:
        print(f"[info] cache_dir={cache_dir}")

    # Download (if needed) and load from HuggingFace
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    model = WavLMForXVector.from_pretrained(
        args.model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    # Save local copy to the requested directory
    feature_extractor.save_pretrained(str(out_dir))
    model.save_pretrained(str(out_dir))
    print(f"[ok] model files saved to: {out_dir}")

    if args.skip_demo:
        return

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Missing datasets dependency for demo. Install with: pip install datasets, "
            "or rerun with --skip_demo"
        ) from exc

    # Demo exactly like the reference usage
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

    # Load from local folder (ensures saved files are valid)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(out_dir))
    model = WavLMForXVector.from_pretrained(str(out_dir))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    audio = [x["array"] for x in dataset[:2]["audio"]]
    inputs = feature_extractor(audio, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        embeddings = model(**inputs).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    similarity = float(cosine_sim(embeddings[0], embeddings[1]))
    print(f"[demo] cosine_similarity={similarity:.6f}")
    print(f"[demo] threshold={args.threshold:.3f}")
    if similarity < args.threshold:
        print("[demo] Speakers are not the same!")
    else:
        print("[demo] Speakers are likely the same.")


if __name__ == "__main__":
    main()
