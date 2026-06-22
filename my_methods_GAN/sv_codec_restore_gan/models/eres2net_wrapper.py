from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

try:
    from speakerlab.models.eres2net.ERes2Net import ERes2Net
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from speakerlab.models.eres2net.ERes2Net import ERes2Net


def _unwrap(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "embedding_model", "frontend_state"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _load_checkpoint(model_path: str | Path):
    model_path = Path(model_path).resolve()
    header = model_path.read_text(encoding="utf-8", errors="ignore")[:128]
    if header.startswith("version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"ERes2Net checkpoint is still a Git LFS pointer, not the real weight file: {model_path}"
        )
    return torch.load(str(model_path), map_location="cpu")


def _infer_m_channels(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["conv1.weight"].shape[0])


def _infer_embedding_dim(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["seg_1.weight"].shape[0])


def _infer_feat_dim(state_dict: dict[str, torch.Tensor], m_channels: int) -> int:
    seg_in = int(state_dict["seg_1.weight"].shape[1])
    return seg_in // (m_channels * 4)


class FrozenERes2Net(torch.nn.Module):
    def __init__(self, model_path: str | Path):
        super().__init__()
        state = _load_checkpoint(model_path)
        state_dict = _unwrap(state)
        m_channels = _infer_m_channels(state_dict)
        feat_dim = _infer_feat_dim(state_dict, m_channels)
        embedding_size = _infer_embedding_dim(state_dict)
        self.embedding_dim = embedding_size
        self.model = ERes2Net(
            feat_dim=feat_dim,
            embedding_size=embedding_size,
            m_channels=m_channels,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, feat_btf: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(feat_btf), dim=-1)
