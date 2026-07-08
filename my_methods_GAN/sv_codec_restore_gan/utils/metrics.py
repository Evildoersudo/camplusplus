from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from speakerlab.utils.score_metrics import compute_c_norm, compute_eer, compute_pmiss_pfa_rbst


def cosine_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1)


def compute_eer_mindcf(labels: list[int], scores: list[float], p_target: float = 0.01) -> tuple[float, float]:
    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    fnr, fpr = compute_pmiss_pfa_rbst(scores_np, labels_np)
    eer, _ = compute_eer(fnr, fpr, scores_np)
    min_dcf = compute_c_norm(fnr, fpr, p_target=p_target, c_miss=1.0, c_fa=1.0)
    return float(100.0 * eer), float(min_dcf)
