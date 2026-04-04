from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class WavLMFeatureExtractor(torch.nn.Module):
    def __init__(self, wavlm_root: str | Path, checkpoint_path: str | Path):
        super().__init__()
        wavlm_root = Path(wavlm_root).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()

        required_files = ["WavLM.py", "modules.py"]
        missing = [name for name in required_files if not (wavlm_root / name).is_file()]
        if missing:
            missing_text = ", ".join(missing)
            raise ImportError(
                f"Missing WavLM source files in {wavlm_root}: {missing_text}. "
                "Download them from https://github.com/microsoft/unilm/tree/master/wavlm "
                "or run:\n"
                f"  curl -fsSL https://raw.githubusercontent.com/microsoft/unilm/master/wavlm/WavLM.py -o {wavlm_root / 'WavLM.py'}\n"
                f"  curl -fsSL https://raw.githubusercontent.com/microsoft/unilm/master/wavlm/modules.py -o {wavlm_root / 'modules.py'}"
            )

        if str(wavlm_root) not in sys.path:
            sys.path.insert(0, str(wavlm_root))

        try:
            from WavLM import WavLM, WavLMConfig
        except Exception as exc:  # pragma: no cover - depends on local WavLM files
            raise ImportError(
                f"Cannot import WavLM from {wavlm_root}. Ensure WavLM.py exists there."
            ) from exc

        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        cfg = WavLMConfig(ckpt["cfg"])
        model = WavLM(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.cfg = cfg

    def forward(self, wav16k: torch.Tensor) -> torch.Tensor:
        x = wav16k
        if getattr(self.cfg, "normalize", False):
            x = F.layer_norm(x, x.shape[-1:])
        rep = self.model.extract_features(x)[0]  # [B, T, D]
        return rep
