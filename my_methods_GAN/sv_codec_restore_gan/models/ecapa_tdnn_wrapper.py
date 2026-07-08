from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

try:
    from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from speakerlab.models.ecapa_tdnn.ECAPA_TDNN import ECAPA_TDNN


def _unwrap(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "embedding_model", "frontend_state"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _infer_channels(state_dict: dict[str, torch.Tensor]) -> list[int]:
    return [
        int(state_dict["blocks.0.conv.conv.weight"].shape[0]),
        int(state_dict["blocks.1.tdnn1.conv.conv.weight"].shape[0]),
        int(state_dict["blocks.2.tdnn1.conv.conv.weight"].shape[0]),
        int(state_dict["blocks.3.tdnn1.conv.conv.weight"].shape[0]),
        int(state_dict["mfa.conv.conv.weight"].shape[0]),
    ]


def infer_ecapa_embedding_dim(model_path: str | Path) -> int:
    state = torch.load(str(Path(model_path).resolve()), map_location="cpu")
    state_dict = _unwrap(state)
    weight = state_dict.get("fc.conv.weight")
    if weight is None:
        raise RuntimeError("ECAPA-TDNN checkpoint missing key 'fc.conv.weight'.")
    return int(weight.shape[0])


class FrozenECAPATDNN(torch.nn.Module):
    def __init__(self, model_path: str | Path):
        super().__init__()
        model_path = Path(model_path).resolve()
        state = torch.load(str(model_path), map_location="cpu")
        state_dict = _unwrap(state)
        self.embedding_dim = int(state_dict["fc.conv.weight"].shape[0])
        self.model = ECAPA_TDNN(
            input_size=int(state_dict["blocks.0.conv.conv.weight"].shape[1]),
            lin_neurons=self.embedding_dim,
            channels=_infer_channels(state_dict),
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, feat_btf: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(feat_btf), dim=-1)
