from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from speakerlab.models.campplus.DTDNN import CAMPPlus


def _unwrap(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "embedding_model", "frontend_state"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _infer_emb_dim(state_dict: dict) -> int:
    weight = state_dict.get("xvector.dense.linear.weight")
    if weight is None:
        return 192
    return int(weight.shape[0])


class FrozenCampPlus(torch.nn.Module):
    def __init__(self, model_path: str | Path):
        super().__init__()
        state = torch.load(str(Path(model_path).resolve()), map_location="cpu")
        state_dict = _unwrap(state)
        self.model = CAMPPlus(feat_dim=80, embedding_size=_infer_emb_dim(state_dict))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, feat_bt80: torch.Tensor) -> torch.Tensor:
        emb = self.model(feat_bt80)
        return F.normalize(emb, dim=-1)
