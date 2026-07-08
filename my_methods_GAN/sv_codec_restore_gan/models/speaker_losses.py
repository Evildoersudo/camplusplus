from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AMSoftmaxClassifier(nn.Module):
    """Additive-margin softmax head for speaker classification."""

    def __init__(self, in_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.margin = float(margin)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = F.normalize(emb, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        cosine = F.linear(x, w)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.scale * (cosine - one_hot * self.margin)
        return F.cross_entropy(logits, labels)
