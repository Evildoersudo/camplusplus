from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

try:
    from speakerlab.models.campplus.DTDNN import CAMPPlus
except ModuleNotFoundError:
    # Allow running from subfolders without manually exporting PYTHONPATH.
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
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
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            missing = ", ".join(incompatible.missing_keys[:8])
            unexpected = ", ".join(incompatible.unexpected_keys[:8])
            raise RuntimeError(
                "CAMPPlus checkpoint is not fully compatible. "
                f"missing_keys=[{missing}] unexpected_keys=[{unexpected}]"
            )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, feat_bt80: torch.Tensor) -> torch.Tensor:
        emb = self.model(feat_bt80)
        return F.normalize(emb, dim=-1)
